import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy
import os
import requests
import json
import io
import base64
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, set_peft_model_state_dict
from torch.utils.data import DataLoader, Subset
from loss import EnhancedVACCLoss
from sql_dataset import TrainDataset
from sklearn.metrics import f1_score, roc_auc_score, classification_report
import time
from datetime import datetime
import psutil


class Llama4SQA_prompt_client(nn.Module):
    def __init__(self, llama_lora_model, num_labels=9, projection_dim=4096):
        super().__init__()

        self.hidden_size = llama_lora_model.config.hidden_size
        self.llama = llama_lora_model

        self.projection_layer = nn.Linear(self.hidden_size, projection_dim)

        self.classifiers = nn.ModuleList([
            nn.Linear(projection_dim, 1) for _ in range(num_labels)
        ])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.projection_layer.weight)
        if self.projection_layer.bias is not None:
            nn.init.zeros_(self.projection_layer.bias)
        for classifier in self.classifiers:
            nn.init.xavier_normal_(classifier.weight)
            if classifier.bias is not None:
                nn.init.zeros_(classifier.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        last_hidden = outputs.hidden_states[-1]
        attention_mask_expanded = attention_mask.unsqueeze(-1)
        masked_hidden = last_hidden * attention_mask_expanded
        sum_hidden = masked_hidden.sum(dim=1)
        valid_token_counts = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_hidden / valid_token_counts

        projected_features = self.projection_layer(pooled.to(self.projection_layer.weight.dtype))
        logits = torch.cat([
            classifier(projected_features) for classifier in self.classifiers
        ], dim=-1)

        return logits


def load_stu_model(args, device, load_path=None, merge_lora=False):
    model_path = args.pre_path + '/models/llama-1B/'
    tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none"
    )

    print("Optimizing for CPU execution, loading Llama model in bfloat16 precision mode...")

    if device == 'cuda:0':
        llama = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device
        )
        print("✓ Llama model loaded with bfloat16 precision on CPU")
    else:
        llama = AutoModelForCausalLM.from_pretrained(model_path)
        print(f"✓ Llama model loaded on device {device}")

    llama_lora = get_peft_model(llama, lora_config)

    student_model = Llama4SQA_prompt_client(llama_lora, num_labels=9, projection_dim=4096).to(device)

    if device == 'cuda:0':
        print("Converting entire client model to bfloat16 precision...")
        student_model = student_model.to(torch.bfloat16)
        print("✓ Model precision reduced to bfloat16")

    if load_path:
        print(f"--- Loading weights for client from {load_path} ---")
        if not os.path.exists(load_path):
            print(f"❌ Error: Weight file {load_path} not found")
            return student_model, tokenizer

        checkpoint = torch.load(load_path, map_location='cpu', weights_only=True)

        if 'lora_weights' in checkpoint:
            lora_weights = checkpoint['lora_weights']
            if device == 'cuda:0':
                lora_weights = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                                for k, v in lora_weights.items()}
            set_peft_model_state_dict(student_model.llama, lora_weights)
            print("✓ LoRA weights loaded successfully.")

        if 'projection_weights' in checkpoint:
            projection_weights = checkpoint['projection_weights']
            if device == 'cuda:0':
                projection_weights = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                                      for k, v in projection_weights.items()}
            student_model.load_state_dict(projection_weights, strict=False)
            print("✓ Adapter layer weights loaded successfully.")

    if merge_lora:
        print("--- Merging LoRA weights for client... ---")
        student_model.llama = student_model.llama.merge_and_unload()
        print("✓ Client LoRA weights merged.")

    return student_model, tokenizer


def serialize_tensor(tensor):
    if tensor.dtype == torch.bfloat16:
        print(f"Converting {tensor.dtype} tensor to float32 for transmission to server")
        tensor_to_save = tensor.float()
    else:
        tensor_to_save = tensor

    if not torch.isfinite(tensor_to_save).all():
        n_nan = torch.isnan(tensor_to_save).sum().item()
        n_inf = torch.isinf(tensor_to_save).sum().item()
        raise ValueError(
            f"Attempted to serialize tensor containing nan={n_nan} / inf={n_inf}, transmission aborted."
        )

    buffer = io.BytesIO()
    torch.save(tensor_to_save, buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def deserialize_tensor(b64_string, device):
    buffer = io.BytesIO(base64.b64decode(b64_string))
    tensor = torch.load(buffer, map_location='cpu')
    tensor = tensor.to(device)
    return tensor


def serialize_state_dict(state_dict):
    print("Preparing weight deltas for transmission to server...")
    serialized_dict = {}
    for k, v in state_dict.items():
        if v.dtype == torch.bfloat16:
            print(f"Converting weight {k} from bfloat16 to float32")
        serialized_dict[k] = serialize_tensor(v)
    print("✓ All weights prepared in float32 format")
    return serialized_dict


def deserialize_state_dict(serialized, device):
    return {k: deserialize_tensor(v, device) for k, v in serialized.items()}


def calculate_metrics(all_pred_labels, all_opt_labels, device='cpu'):
    all_pred_labels = all_pred_labels.to(device)
    all_opt_labels = all_opt_labels.to(device)

    y_pred = all_pred_labels.cpu().numpy().flatten()
    y_true = (all_opt_labels > 0.1).float().cpu().numpy().flatten()

    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    try:
        auc_micro = roc_auc_score(y_true, y_pred, average='micro')
        auc_macro = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        auc_micro = 0.0
        auc_macro = 0.0
        print("Unable to compute AUC, possibly due to single-class issue")

    # Subset Accuracy
    all_opt_labels_bool = (all_opt_labels > 0.1).float()
    subset_accuracy = torch.all(all_pred_labels == all_opt_labels_bool, dim=1).float().mean()

    # R-ACC
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    precision_1 = report_dict.get('1.0', {}).get('precision', 0.0)
    print(f"R-ACC: {precision_1:.4f}")

    return {
        'F1_micro': f1_micro,
        'F1_macro': f1_macro,
        'AUC_micro': auc_micro,
        'AUC_macro': auc_macro,
        'Subset_Accuracy': subset_accuracy.item()
    }


class FederatedClient:
    def __init__(self, args, client_id, server_url):
        self.args = args
        self.device = args.device
        self.client_id = client_id
        self.server_url = server_url
        self.communication_times = []
        print(f"--- Initializing client {client_id} ---")
        client_path = args.client_path
        self.model, self.tokenizer = load_stu_model(
            args, self.device, load_path=client_path, merge_lora=True
        )

        self.sync_classifier_weights()
        print(f"✓ Client {client_id} initialization complete")

    def sync_classifier_weights(self):
        print(f"Client {self.client_id} syncing classifier head from server...")
        try:
            response = requests.get(f"{self.server_url}/get_classifier_weights")
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    classifier_weights = deserialize_state_dict(data['classifier_weights'], self.device)

                    model_dtype = next(self.model.parameters()).dtype
                    print(f"Client model dtype: {model_dtype}")

                    if model_dtype == torch.bfloat16:
                        print("Converting server weights from float32 to bfloat16 to match client model...")
                        classifier_weights = {
                            k: v.to(torch.bfloat16) if v.dtype in [torch.float32, torch.float64] else v
                            for k, v in classifier_weights.items()
                        }
                        print("✓ Server weights converted to bfloat16")
                    elif model_dtype == torch.float32:
                        print("Server weights are already float32, no conversion needed")
                        classifier_weights = {
                            k: v.float() if v.dtype == torch.bfloat16 else v
                            for k, v in classifier_weights.items()
                        }

                    self.model.classifiers.load_state_dict(classifier_weights)
                    print(f"✓ Client {self.client_id} synced latest classifier head weights")
                    return True
            else:
                print(f"❌ Server response error, status code: {response.status_code}")
        except Exception as e:
            print(f"❌ Error during weight sync: {str(e)}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
        return False

    def get_data_loader(self, data_split='val'):
        """Get data loader for client"""
        full_data = TrainDataset(self.args, self.tokenizer, data_split)
        client_subset = Subset(full_data, range(len(full_data)))
        loader = DataLoader(client_subset, batch_size=self.args.batch_size, shuffle=False)
        return loader

    def evaluate(self):
        """Evaluate current model on local dataset before training (mimic server behavior)"""
        print(f"\n--- Client {self.client_id}: Starting local initial performance evaluation ---")
        eval_loader = self.get_data_loader('test')

        self.model.eval()
        all_pred_probs = []
        all_opt_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Client {self.client_id} evaluating"):
                input_ids, att_mask, labels, opt_labels = [t.to(self.device) for t in batch]

                if self.device == 'cuda:0' and next(self.model.parameters()).dtype == torch.bfloat16:
                    input_ids = input_ids.to(torch.bfloat16) if input_ids.dtype == torch.float32 else input_ids
                    att_mask = att_mask.to(torch.bfloat16) if att_mask.dtype == torch.float32 else att_mask

                logits = self.model(input_ids=input_ids, attention_mask=att_mask)
                probs = torch.sigmoid(logits)

                all_pred_probs.append(probs.float().cpu())
                all_opt_labels.append(opt_labels.float().cpu())

        if not all_pred_probs:
            print("Evaluation data is empty, skipping metric calculation.")
            self.model.train()
            return

        all_pred_probs = torch.cat(all_pred_probs, dim=0)
        all_opt_labels = torch.cat(all_opt_labels, dim=0)
        all_multilabel_pred = (all_pred_probs > 0.5).float()

        print(f"\nClient Model Performance (ID: {self.client_id}):")
        metrics = calculate_metrics(all_multilabel_pred, all_opt_labels, 'cpu')

        print(f"VACC: {metrics['F1_micro']:.4f}")
        print(f"F1: {metrics['F1_macro']:.4f}")
        print(f"AUC: {metrics['AUC_micro']:.4f}")

        self.model.train()

    def train(self, epochs=1, use_cached_batches=False, cache_dir="cached_batches", batch_indices=[0,1,2,3]):
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f"{cache_dir}/client_{self.client_id}_batches.pt"

        all_cached_batches = []
        if use_cached_batches and os.path.exists(cache_file):
            print(f"Loading cached batch data: {cache_file}")
            try:
                cached_data = torch.load(cache_file)
                all_cached_batches = cached_data['selected_batches']
                print(f"✓ Successfully loaded {len(all_cached_batches)} cached batches")

                for i, batch in enumerate(all_cached_batches):
                    print(f"  Batch #{i}: index={batch['batch_idx']}, num_samples={len(batch['sample_indices'])}")

            except Exception as e:
                print(f"❌ Failed to load cache: {e}, regenerating batches")
                use_cached_batches = False
                all_cached_batches = []
        else:
            use_cached_batches = False

        if not use_cached_batches:
            train_loader = self.get_data_loader('val')
            all_samples = list(train_loader.dataset)
            batch_size = 3
            num_batches = (len(all_samples) + batch_size - 1) // batch_size
            all_possible_batches = [
                {'batch_idx': i, 'samples': all_samples[i * batch_size:min((i + 1) * batch_size, len(all_samples))]}
                for i in range(num_batches)
            ]

            num_select = min(4, len(all_possible_batches))
            selected_batch_infos = random.sample(all_possible_batches, num_select)
            print(f"Randomly selected {num_select} batches for training")

            batches_to_save = []
            for batch in selected_batch_infos:
                sample_indices = list(range(
                    batch['batch_idx'] * batch_size,
                    min((batch['batch_idx'] + 1) * batch_size, len(all_samples))
                ))
                batches_to_save.append({
                    'batch_idx': batch['batch_idx'],
                    'sample_indices': sample_indices
                })

            try:
                torch.save({'selected_batches': batches_to_save}, cache_file)
                print(f"✓ Batch indices cached to {cache_file}")
                all_cached_batches = batches_to_save

                for i, batch in enumerate(all_cached_batches):
                    print(f"  Batch #{i}: index={batch['batch_idx']}, num_samples={len(batch['sample_indices'])}")

            except Exception as e:
                print(f"❌ Failed to save batch cache: {e}")

        if batch_indices is not None and all_cached_batches:
            valid_indices = [i for i in batch_indices if 0 <= i < len(all_cached_batches)]
            if len(valid_indices) != len(batch_indices):
                print(f"⚠️ Warning: Some batch indices are invalid. Valid range: 0-{len(all_cached_batches) - 1}")
            selected_batch_infos = [all_cached_batches[i] for i in valid_indices]
            print(f"✓ Selected {len(selected_batch_infos)} specified batches (indices: {valid_indices})")
        else:
            selected_batch_infos = all_cached_batches if all_cached_batches else batches_to_save

        train_loader = self.get_data_loader('val')
        all_samples = list(train_loader.dataset)
        selected_batches = []

        for batch_info in selected_batch_infos:
            try:
                samples = [all_samples[idx] for idx in batch_info['sample_indices']]
                selected_batches.append({
                    'batch_idx': batch_info['batch_idx'],
                    'samples': samples
                })
            except IndexError as e:
                print(f"❌ Index error in batch {batch_info['batch_idx']}: {e}")
                continue

        if not selected_batches:
            print("❌ Error: No valid batches available for training!")
            return None

        initial_weights = copy.deepcopy(self.model.classifiers.state_dict())

        self.model.train()
        for name, param in self.model.named_parameters():
            param.requires_grad = "classifiers" in name

        loss_fn = EnhancedVACCLoss(epsilon=0.1, alpha=1, beta=0, gamma=0).to(self.device)

        selected_deltas = []
        process = psutil.Process()
        cpu_start = process.cpu_percent()

        print(f"Starting training...")
        start_time = time.time()
        for batch_idx, batch in enumerate(tqdm(selected_batches, desc=f"Client {self.client_id} training")):
            print(f"Training batch {batch_idx} (original index: {batch['batch_idx']}, num_samples: {len(batch['samples'])})")

            current_weights = copy.deepcopy(self.model.classifiers.state_dict())

            for sample in batch['samples']:
                input_ids, att_mask, labels, opt_labels = [t.unsqueeze(0).to(self.device) for t in sample]
                optimizer = torch.optim.AdamW(self.model.classifiers.parameters(), lr=1e-5)

                if self.device == 'cuda:0' and next(self.model.parameters()).dtype == torch.bfloat16:
                    input_ids = input_ids.to(torch.bfloat16) if input_ids.dtype == torch.float32 else input_ids
                    att_mask = att_mask.to(torch.bfloat16) if att_mask.dtype == torch.float32 else att_mask
                    labels = labels.to(torch.bfloat16) if labels.dtype == torch.float32 else labels
                    opt_labels = opt_labels.to(torch.bfloat16) if opt_labels.dtype == torch.float32 else opt_labels

                logits = self.model(input_ids=input_ids, attention_mask=att_mask)
                loss, _, _, _ = loss_fn(logits, labels, opt_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            trained_weights = self.model.classifiers.state_dict()
            weight_delta = {key: trained_weights[key] - current_weights[key] for key in trained_weights}
            selected_deltas.append(weight_delta)
            self.model.classifiers.load_state_dict(current_weights)

        end_time = time.time()
        cpu_end = process.cpu_percent()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        total_memory = psutil.virtual_memory().total / (1024 * 1024)
        memory_percent = (process.memory_info().rss / psutil.virtual_memory().total) * 100

        duration = end_time - start_time
        print(f"Training completed - Duration: {duration:.2f}s, CPU usage: {cpu_end:.1f}%")
        print(f"Memory usage: {memory_mb:.1f}MB ({memory_percent:.1f}%)")
        print(f"Total system memory: {total_memory:.0f}MB, Process usage: {memory_mb:.1f}MB ({memory_percent:.1f}%)")

        final_weight_delta = {}
        for key in initial_weights:
            deltas_tensor = torch.stack([d[key].float().to(self.device) for d in selected_deltas])
            final_weight_delta[key] = deltas_tensor.mean(dim=0).cpu()

        print(f"Done! Computed average delta over {len(selected_batches)} batches")
        self.model.classifiers.load_state_dict(initial_weights)
        return final_weight_delta

    def submit_delta(self, delta):
        try:
            total_nan = 0
            total_inf = 0
            for name, tensor in delta.items():
                total_nan += torch.isnan(tensor).sum().item()
                total_inf += torch.isinf(tensor).sum().item()

            if total_nan + total_inf > 0:
                print(f"⚠️ Client {self.client_id} delta contains nan={total_nan}, inf={total_inf}, submission aborted")
                return None

            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Client {self.client_id} submitting weight delta...")

            serialize_start = time.time()
            serialized_delta = serialize_state_dict(delta)
            serialize_time = time.time() - serialize_start

            total_size = sum(len(v.encode('utf-8')) for v in serialized_delta.values())
            size_mb = total_size / (1024 * 1024)

            print(f"  - Serialization completed, duration: {serialize_time:.3f}s")
            print(f"  - Data size: {size_mb:.2f}MB")

            send_timestamp = time.time()
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Starting network transmission...")

            response = requests.post(
                f"{self.server_url}/submit_delta",
                json={
                    'client_id': self.client_id,
                    'delta': serialized_delta,
                    'send_timestamp': send_timestamp
                },
                timeout=(10, 120)
            )

            receive_timestamp = time.time()
            data = response.json()
            eval_time = data.get('eval_time', 0)
            network_time = receive_timestamp - send_timestamp - eval_time
            total_time = receive_timestamp - serialize_start - eval_time

            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Network transmission completed")
            print(f"  - Network time: {network_time:.3f}s")
            print(f"  - Transfer speed: {size_mb / network_time:.2f}MB/s" if network_time > 0.001 else "Transfer speed: >1000MB/s")
            print(f"  - Total time (serialize + transmit): {total_time:.3f}s")

            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    print(f"✓ Client {self.client_id} delta submitted successfully")

                    comm_stats = {
                        'round': len(self.communication_times) + 1,
                        'client_id': self.client_id,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                        'serialize_time': serialize_time,
                        'network_time': network_time,
                        'total_time': total_time,
                        'data_size_mb': size_mb,
                        'transfer_speed_mbps': size_mb / network_time if network_time > 0.001 else 999.0,
                        'round_completed': data.get('round_completed', False),
                        'server_network_time': data.get('communication_time', 0),
                        'server_data_size_mb': data.get('data_size_mb', 0)
                    }
                    self.communication_times.append(comm_stats)
                    self._save_communication_times()

                    if data.get('round_completed', False):
                        self._print_communication_summary()

                    return data
                else:
                    print(f"❌ Server returned error: {data}")
            else:
                print(f"❌ Server request failed, status code: {response.status_code}")

        except requests.exceptions.ConnectTimeout:
            print(f"❌ Connection to server timed out, check if server is running")
        except requests.exceptions.ReadTimeout:
            print(f"❌ Server response timed out, possibly processing large data, retry later")
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to server {self.server_url}, check network connection")
        except requests.exceptions.RequestException as e:
            print(f"❌ Network exception: {e}")
        except Exception as e:
            print(f"❌ Unknown error during delta submission: {e}")

        return None

    def _save_communication_times(self):
        """Save communication times to file"""
        import json
        filename = f"client_{self.client_id}_communication_times.json"
        try:
            with open(filename, 'w') as f:
                json.dump(self.communication_times, f, indent=2)
            print(f"✓ Communication times saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save communication times: {e}")

    def _print_communication_summary(self):
        """Print communication time summary"""
        if not self.communication_times:
            return

        print(f"\n=== Client {self.client_id} Communication Summary ===")

        serialize_times = [t['serialize_time'] for t in self.communication_times]
        network_times = [t['network_time'] for t in self.communication_times]
        total_times = [t['total_time'] for t in self.communication_times]
        data_sizes = [t['data_size_mb'] for t in self.communication_times]
        speeds = [t['transfer_speed_mbps'] for t in self.communication_times if t['transfer_speed_mbps'] < 999]

        print(f"Total rounds: {len(self.communication_times)}")
        print(f"Serialization time - Avg: {np.mean(serialize_times):.3f}s, Max: {np.max(serialize_times):.3f}s")
        print(f"Network time - Avg: {np.mean(network_times):.3f}s, Max: {np.max(network_times):.3f}s")
        print(f"Total time - Avg: {np.mean(total_times):.3f}s, Max: {np.max(total_times):.3f}s")
        print(f"Data size - Avg: {np.mean(data_sizes):.2f}MB")
        if speeds:
            print(f"Transfer speed - Avg: {np.mean(speeds):.2f}MB/s")
        print("=" * 50)

    def federated_round(self):
        """Execute one federated learning round"""
        print(f"\n--- Client {self.client_id} starting local training ---")
        delta = self.train(epochs=1)
        result = self.submit_delta(delta)

        if result and result.get('round_completed', False):
            print(f"\n--- Federated round {result.get('round_num', '?')} completed ---")
            metrics = result.get('metrics', {})
            if metrics:
                print(f"Global model performance update:")
                for k, v in metrics.items():
                    print(f"  {k}: {v:.4f}")
                metrics_file = f"client_{self.client_id}_metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=4)
                print(f"✓ Metrics saved to {metrics_file}")
            self.sync_classifier_weights()
            return True

        return False


def main_client(args, client_id, server_url, num_rounds=5):
    """Main function for client"""
    torch.manual_seed(args.seed if hasattr(args, 'seed') else 42)
    np.random.seed(args.seed if hasattr(args, 'seed') else 42)

    # Phase 1: Initialize client
    print(f"\n[Phase 1] Initializing client {client_id}...")
    client = FederatedClient(args, client_id, server_url)

    if client_id == 0:
        try:
            requests.post(f"{server_url}/set_num_clients", json={'num_clients': args.num_clients})
            print(f"✓ Notified server of expected number of clients: {args.num_clients}")
        except Exception as e:
            print(f"❌ Error setting number of clients: {str(e)}")

    # Phase 2: Baseline evaluation of initial synchronized model
    print(f"\n[Phase 2] Performing baseline evaluation on initially synced model...")
    try:
        client.evaluate()
    except Exception as e:
        print(f"❌ Critical error during initial model evaluation: {e}")

    # Phase 3: Execute multiple federated rounds
    print(f"\n[Phase 3] Starting federated learning rounds...")
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'=' * 15} Client {client_id} | Round {round_num}/{num_rounds} {'=' * 15}")
        round_completed = client.federated_round()

        if round_completed:
            print(f"✓ Round {round_num} completed")
        else:
            print(f"⚠️ Round {round_num} not completed or waiting for aggregation, proceeding to next round")

    print(f"\nClient {client_id} federated learning completed!")