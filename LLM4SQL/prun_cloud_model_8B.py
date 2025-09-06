import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import gc
import hashlib
import time
import os
import pandas as pd
from typing import List, Tuple

# ---------- Project Core Dependencies ----------
from loss import EnhancedVACCLoss
from Embedding import PlanGAT, TimeLSTM
from CombinedDataset import CombinedDataset
from Combine_model import CombinedModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from sql_dataset import TrainDataset
from subdateset import CustomSubset
from model import Llama4SQA
from model_prompt import Llama4SQA_prompt
from sklearn.metrics import f1_score, jaccard_score, roc_auc_score, classification_report
from nltk.tokenize import word_tokenize
import nltk
from torch_geometric.data import Data, Batch
from copy import deepcopy

# --- Part 1: Auxiliary Functions and Model Class Definitions ---

def load_text_model(vocab_size, embedding_dim, load_path, device):
    """Load saved text_model (nn.Embedding), strictly following your definition."""
    text_model = nn.Embedding(vocab_size, embedding_dim).to(device)
    try:
        text_model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))
        text_model.eval()
        print(f"Successfully loaded text embedding model: {load_path}")
    except FileNotFoundError:
        print(f"Warning: Text embedding model file not found at {load_path}. Using randomly initialized embedding layer!")
    return text_model


class InferenceModel(nn.Module):
    """
    Integrated final inference model.
    It contains the merged LoRA LLM and all other pre-trained components.
    """

    def __init__(self, merged_llm_model, graph_dim, time_dim, fusion_dim, text_dim, num_labels=9):
        super().__init__()

        self.llm = merged_llm_model
        self.graph_model = PlanGAT(in_channels=11, hidden_channels=128, out_channels=graph_dim)
        self.time_model = TimeLSTM(input_size=1, hidden_size=64, q=7, output_dim=40, num_layers=1)

        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.graph_proj = nn.Linear(graph_dim, fusion_dim)
        self.time_proj = nn.Linear(280, fusion_dim)  # TimeLSTM output dimension is 280

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(fusion_dim // 2, fusion_dim)
        )

        self.classifiers = nn.ModuleList([
            nn.Linear(self.llm.config.hidden_size, 1) for _ in range(num_labels)
        ])

    def forward(self, text_inputs, graph_data, time_series):
        text_features = self.text_proj(text_inputs)
        graph_embedding = self.graph_model(graph_data)
        graph_features = self.graph_proj(graph_embedding)
        time_embedding = self.time_model(time_series)
        time_embedding = time_embedding.squeeze(0)
        time_features = self.time_proj(time_embedding)

        fused_features = text_features + graph_features + time_features
        fusion_embeding = self.fusion_layer(fused_features)

        llm_input_embeds = fusion_embeding.unsqueeze(0).unsqueeze(1)
        # Regardless of previous operations, ensure type and device are correct before entering LLM.
        llm_input_embeds = llm_input_embeds.to(device=self.llm.device, dtype=self.llm.dtype)
        outputs = self.llm(inputs_embeds=llm_input_embeds,
                           output_hidden_states=True,
                           return_dict=True)
        llm_features = outputs.hidden_states[-1].mean(dim=1)

        logits = torch.cat([classifier(llm_features) for classifier in self.classifiers], dim=1)
        return logits


def hash_word(word, vocab_size):
    # Use MD5 hash, take first 8 hex digits converted to integer, ensure fixed result
    hash_obj = hashlib.md5(word.encode('utf-8'))
    hash_int = int(hash_obj.hexdigest()[:8], 16)  # Take first 8 hex digits
    return (hash_int % (vocab_size - 1)) + 1  # Map to vocabulary range


def text_to_vector(text, model, args, vector_size=64):
    """
    Convert text to vector representation (fix device mismatch issue)
    """
    # Initialize stopword filtering (optional)
    nltk.data.path.append(args.pre_path + '/models/nltk_data')

    # Check if input is a tuple
    if isinstance(text, tuple):
        if len(text) > 0:
            text = text[0]
        else:
            print("Using all-zero prompt")
            return torch.zeros(vector_size, device=args.device)

    # Ensure text is string type
    if not isinstance(text, str):
        text = str(text)

    # Tokenization
    tokens = word_tokenize(text.lower())

    # Convert words to hash indices
    indices = [hash_word(word, model.num_embeddings) for word in tokens]

    if not indices:
        return torch.zeros(vector_size, device=args.device)

    # Create tensor and ensure it's on the same device as model
    indices_tensor = torch.tensor(indices, dtype=torch.long, device=model.weight.device)

    # Use embedding layer to get vectors
    vectors = model(indices_tensor)

    # Compute average vector
    return torch.mean(vectors, dim=0)


def calculate_metrics(all_pred_labels, all_opt_labels, device='cuda:0'):
    """
    Calculate evaluation metrics including F1, AUC, and Subset Accuracy

    Parameters:
    all_pred_labels (torch.Tensor): Predicted labels
    all_opt_labels (torch.Tensor): True labels
    device (str): Computation device, default is cuda:0

    Returns:
    dict: Dictionary containing various evaluation metrics
    """
    # Ensure all tensors are on the same device
    all_pred_labels = all_pred_labels.to(device)
    all_opt_labels = all_opt_labels.to(device)

    # Convert to numpy arrays
    y_pred = all_pred_labels.cpu().numpy().flatten()  # Move to CPU before conversion
    y_true = (all_opt_labels > 0.1).float().cpu().numpy().flatten()  # Move to CPU before conversion

    # Calculate F1 score (micro and macro)
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    # Calculate AUC (micro and macro)
    try:
        auc_micro = roc_auc_score(y_true, y_pred, average='micro')
        auc_macro = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        # Handle case with only one class
        auc_micro = 0.0
        auc_macro = 0.0
        print("AUC calculation failed, possibly due to single class scenario")

    # Calculate Subset Accuracy
    subset_accuracy = torch.all(all_pred_labels == (all_opt_labels > 0.1).float(), dim=1).float().mean()
    print(classification_report(y_true, y_pred, zero_division=0))
    return {
        'F1_micro': f1_micro,
        'F1_macro': f1_macro,
        'AUC_micro': auc_micro,
        'AUC_macro': auc_macro,
        'Subset_Accuracy': subset_accuracy.item()
    }


def cloud_pruning_experiment_8B(args, snip_sparsity, drop_layers_percent, heads_prune_percent):
    """
    Final fixed version: Run a complete pruning and testing experiment.
    """
    device = args.device
    print("\n" + "=" * 80)
    print(f"== Starting New Experiment: SNIP Sparsity={snip_sparsity:.2f}, Layer Prune={drop_layers_percent:.2%}, Head Prune={heads_prune_percent:.2%} ==")
    print("=" * 80)

    # --- Part 1: Model Loading and Merging ---
    print("【Part A】Starting to build complete inference model in memory...")
    # (This part of code remains unchanged)
    base_model_path = args.pre_path + '/models/LLAMA3.1-8B-Instruct/'
    model_file_base_name = args.cloud2prun
    trained_checkpoint_path = f"{args.pre_path}/combined_model/{model_file_base_name}.pth"
    text_model_path = f"{args.pre_path}/combined_model/{model_file_base_name.replace('llama_8B', 'text_llama_8B')}.pth"
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, legacy=False)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False,
        r=32, lora_alpha=64, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none"
    )

    model_with_lora_scaffold = PeftModel(base_model, lora_config)

    print("Loading checkpoint...")
    trained_checkpoint = torch.load(trained_checkpoint_path, map_location='cpu', weights_only=True)
    combined_model_state_dict = trained_checkpoint['combined_model_state']

    lora_prefix = 'LLM_model.llama.'
    lora_weights = {k[len(lora_prefix):]: v for k, v in combined_model_state_dict.items() if k.startswith(lora_prefix)}
    model_with_lora_scaffold.load_state_dict(lora_weights, strict=True)

    print("Merging LoRA weights...")
    merged_llm_model = model_with_lora_scaffold.merge_and_unload()
    print("LLM LoRA merging completed.")

    del base_model, model_with_lora_scaffold, lora_weights
    gc.collect()
    torch.cuda.empty_cache()

    for param in merged_llm_model.parameters():
        param.requires_grad = True
    merged_llm_model.train()
    report_sparsity(merged_llm_model, prefix="[After Base Merging Model] ")

    # --- Part 2: Pruning ---
    train_data_prue = TrainDataset(args, tokenizer, 'train')
    dataloader_train_prue = DataLoader(train_data_prue, batch_size=args.batch_size, shuffle=True, num_workers=0)

    if snip_sparsity > 0:
        print("\n--- Performing Unstructured Pruning (SNIP) ---")
        # Call the ultimate memory-efficient pruning function
        snip_prune_with_mask_ultimate(
            model=merged_llm_model,
            data_loader=dataloader_train_prue,
            device=device,
            sparsity=snip_sparsity
        )
        report_sparsity(merged_llm_model, prefix="[After SNIP Pruning] ")

    print("\n--- Performing Structured Pruning ---")
    teacher_model = apply_structured_pruning(
        merged_llm_model,
        dataloader_train_prue,
        device,
        drop_layers_percent=drop_layers_percent,
        heads_prune_percent=heads_prune_percent
    )
    final_total_params, final_nonzero_params = report_sparsity(teacher_model, prefix="[Final Pruned LLM] ")
    teacher_model = merged_llm_model
    del merged_llm_model, dataloader_train_prue, train_data_prue
    gc.collect()
    torch.cuda.empty_cache()

    # --- Part 3: Assemble Final Inference Model ---
    print("\nCreating final model framework...")
    final_inference_model = InferenceModel(
        merged_llm_model=teacher_model,
        graph_dim=256, time_dim=280, fusion_dim=4096, text_dim=64
    ).to(device)

    print("Loading weights for Graph/Time/Fusion/Classifier layers...")
    non_lora_weights = {}
    # Now combined_model_state_dict is available, no error will be raised
    for key, value in combined_model_state_dict.items():
        if not key.startswith(lora_prefix):
            new_key = key.replace('LLM_model.', '') if key.startswith('LLM_model.classifiers.') else key
            non_lora_weights[new_key] = value

    final_inference_model.load_state_dict(non_lora_weights, strict=False)
    print("All component weights loaded successfully.")

    # Clean up final large objects
    del combined_model_state_dict, non_lora_weights, trained_checkpoint
    gc.collect()

    text_model = load_text_model(
        vocab_size=10000,
        embedding_dim=64,
        load_path=text_model_path,
        device=device
    )
    print("Complete inference model is ready!")

    # ==================== PART B: Evaluate Model on Test Set ====================
    print("\n【Part B】Starting model evaluation on test set...")
    test_data = CombinedDataset(args, 'test')
    dataloader_test = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    final_inference_model.eval()
    # Get model weight data type (usually bfloat16)
    model_dtype = torch.bfloat16
    final_inference_model = final_inference_model.to(device).to(dtype=model_dtype)  # Move to device first, then change dtype
    test_len = len(dataloader_test.dataset)

    with torch.no_grad():
        top_1_cor = 0
        rank_pred = []
        rank_true = []
        top1_valid_sum = 0
        all_right_cnt = 0
        all_pred_opt = []
        all_opt_labels = []
        # <<< MODIFICATION START: Initialize inference timer >>>
        total_inference_time = 0.0
        # Use CUDA events for precise GPU timing
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for batch_idx, batch in enumerate(tqdm(dataloader_test, leave=True)):
            prompt, labels, opt_labels, graph_data, time_series = batch
            # Process text data
            text_embeddings = text_to_vector(prompt, text_model, args)
            text_embeddings = text_embeddings.to(args.device)

            labels, opt_labels = labels.to(args.device), opt_labels.to(args.device)
            # Process graph data
            if isinstance(graph_data, dict):
                # Check if manual wrapping is needed
                if "batch" not in graph_data:  # If not already a batched object
                    x = graph_data["x"].to(args.device)
                    edge_index = graph_data["edge_index"].to(args.device)

                    # Check and remove extra batch dimension (if exists)
                    if x.dim() == 3 and x.shape[0] == 1:
                        x = x.squeeze(0)  # From [1, num_nodes, feature_dim] to [num_nodes, feature_dim]
                    if edge_index.dim() == 3 and edge_index.shape[0] == 1:
                        edge_index = edge_index.squeeze(0)  # From [1, num_nodes, feature_dim] to [num_nodes, feature_dim]
                    # Create Data object
                    data = Data(x=x, edge_index=edge_index)

                    # Wrap into batch (only if needed)
                    graph_batch = Batch.from_data_list([data])
                else:
                    # If graph_data is already a batch object
                    graph_batch = graph_data.to(args.device)
            else:
                graph_batch = graph_data.to(args.device)

            # Adjust time series device
            time_series = time_series.to(args.device)
            # Forward pass
            text_embeddings = text_embeddings.to("cuda:0")
            # Convert all input tensors to model-consistent dtype
            text_embeddings = text_embeddings.to(dtype=model_dtype)
            # Convert time series dtype
            time_series = time_series.to(dtype=model_dtype)
            graph_batch.x = graph_batch.x.to(dtype=model_dtype)
            # <<< MODIFICATION START: Measure single inference time >>>
            starter.record()
            logits = final_inference_model(
                text_inputs=text_embeddings,  # Directly pass embeddings
                graph_data=graph_batch,
                time_series=time_series
            )
            ender.record()
            torch.cuda.synchronize()  # Wait for GPU tasks to complete
            curr_time = starter.elapsed_time(ender)  # Calculate elapsed time in milliseconds
            total_inference_time += curr_time
            # <<< MODIFICATION END >>>
            probs = torch.sigmoid(logits)
            # probs = logits
            pred_opt = probs
            sorted_time_index = torch.argsort(pred_opt, dim=1, descending=True)
            label_sorted_time_index = torch.argsort(opt_labels, dim=1, descending=True)
            # Move all relevant data to same device
            sorted_time_index = sorted_time_index.to(args.device)
            label_sorted_time_index = label_sorted_time_index.to(args.device)
            for k_i in range(len(pred_opt)):
                p_rank = torch.empty_like(sorted_time_index[k_i])
                p_rank[sorted_time_index[k_i]] = torch.arange(len(pred_opt[k_i])).to(args.device)
                rank_pred.append(p_rank.tolist())

                t_rank = torch.empty_like(label_sorted_time_index[k_i])
                t_rank[label_sorted_time_index[k_i]] = torch.arange(len(opt_labels[k_i])).to(args.device)
                rank_true.append(t_rank.tolist())

                if label_sorted_time_index[k_i][0] == sorted_time_index[k_i][0]: top_1_cor += 1
                if all(label_sorted_time_index[k_i][:3] == sorted_time_index[k_i][:3]): all_right_cnt += 1
                top1_valid_sum += opt_labels[k_i][sorted_time_index[k_i][0]]
            all_pred_opt.append(pred_opt)
            all_opt_labels.append(opt_labels)
        # Merge prediction and ground truth from all batches
        all_pred_opt = torch.cat(all_pred_opt, dim=0)
        all_opt_labels = torch.cat(all_opt_labels, dim=0)
        # Calculate average inference time
        avg_inference_time_ms = total_inference_time / test_len if test_len > 0 else 0
        # Calculate regression metrics
        mse = torch.mean((all_pred_opt - all_opt_labels) ** 2).item()
        mae = torch.mean(torch.abs(all_pred_opt - all_opt_labels)).item()
        # Calculate V-ACC
        all_multilabel_pred = (all_pred_opt > 0.5).float()
        all_multilabel_true = (all_opt_labels > 0.1).float()
        right_label_all = torch.where(all_multilabel_true == all_multilabel_pred, 1, 0).sum()
        vacc = right_label_all / test_len / 9
        top1acc = top_1_cor / float(test_len)
        mcacc = all_right_cnt / float(test_len)
        top1IR = top1_valid_sum / float(test_len)
        print(f'Baseline---Test MSE: {mse}, MAE: {mae}, V-ACC: {vacc},top1acc:{top1acc},mcacc:{mcacc},top1IR:{top1IR}')
        metrics = calculate_metrics(all_multilabel_pred, all_opt_labels)
        print("\nAdditional Evaluation Metrics:")
        print(f"F1 Micro: {metrics['F1_micro']:.4f}")
        print(f"F1 Macro: {metrics['F1_macro']:.4f}")
        print(f"AUC Micro: {metrics['AUC_micro']:.4f}")
        # --- Get GPU Memory ---
        gpu_mem = measure_gpu_memory(final_inference_model, device, note="[Pruned Model]")
        print(f"Inference time {avg_inference_time_ms}")


def measure_gpu_memory(model, device, note=""):
    """
    Measure and print GPU memory usage of the model.
    Args:
        model: The model to measure.
        device: Device where the model resides.
        note (str): Description printed with the result.
    """
    # Move model to correct device and clear cache for cleaner measurement
    model.to(device)
    torch.cuda.empty_cache()

    # Get allocated memory (in MiB)
    allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
    # Get PyTorch reserved memory (in MiB)
    reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 2)

    print(f"{note} GPU Memory Usage -> Allocated: {allocated_memory:.2f} MiB | Reserved: {reserved_memory:.2f} MiB")
    return allocated_memory


# ====================== 1. Structured Pruning Tools ================================
def compute_layer_importance_by_norm(model: nn.Module, norm_type: str = "l1") -> torch.Tensor:
    """
    Evaluate layer importance by computing parameter norms.

    Args:
        model: Model to evaluate (complete AutoModelForCausalLM).
        norm_type: 'l1' or 'l2'.

    Returns:
        A tensor containing importance scores for each Transformer layer.
    """
    layer_importances = []
    # Note: We evaluate layers in stu_model.model.layers
    for layer in model.model.layers:
        layer_norm = 0.0
        for param in layer.parameters():
            if norm_type == "l1":
                layer_norm += torch.sum(torch.abs(param.data))
            elif norm_type == "l2":
                layer_norm += torch.sum(param.data ** 2)
            else:
                raise ValueError("norm_type must be 'l1' or 'l2'")

        if norm_type == "l2":
            layer_norm = torch.sqrt(layer_norm)

        layer_importances.append(layer_norm)

    return torch.tensor(layer_importances, device='cpu')


# --------------------------
# 1. Embedding Column Pruning
# --------------------------
# None currently

# --------------------------
# 2. Transformer Layer Pruning
# --------------------------
def drop_transformer_layers(hf_llama, drop_idx):
    """
    Remove specified index TransformerBlock layers and update model
    """
    kept_layers = []
    for i, layer in enumerate(hf_llama.model.layers):
        if i not in drop_idx:
            kept_layers.append(layer)

    # Update model's layer structure
    hf_llama.model.layers = nn.ModuleList(kept_layers)

    # Update model configuration to ensure correct number of layers
    hf_llama.model.config.num_hidden_layers = len(kept_layers)

    # Clean up redundant parameters: Remove parameters from unused layers
    model_state_dict = hf_llama.state_dict()
    hf_llama.load_state_dict(model_state_dict, strict=False)

    # Update model's parameter count
    hf_llama.num_parameters = sum(p.numel() for p in hf_llama.parameters())
    print(f"Model parameter count after pruning: {hf_llama.num_parameters}")


# --------------------------
# 3. Llama Attention Head Pruning (GQA-Aware, Group-wise Pruning)
# --------------------------
class LlamaAttentionHeadPruner:
    """
    Llama model attention head pruning utility class (supports GQA/MQA, ensures group-wise pruning)
    """

    def __init__(self, model):
        """
        Initialize pruner
        Args:
            model: Llama model instance (usually model.model)
        """
        self.model = model
        self.config = model.config
        self.original_hidden_size = self.config.hidden_size
        self.original_num_q_heads = self.config.num_attention_heads

        # Compatibility with old configurations without num_key_value_heads
        self.original_num_kv_heads = getattr(self.config, 'num_key_value_heads', self.original_num_q_heads)

        self.head_dim = self.original_hidden_size // self.original_num_q_heads

        if self.original_num_q_heads % self.original_num_kv_heads != 0:
            raise ValueError("Q heads must be a multiple of KV heads for GQA.")
        self.num_key_value_groups = self.original_num_q_heads // self.original_num_kv_heads

    def prune_heads(self, kv_heads_to_prune_per_layer: List[Tuple[int, List[int]]]) -> None:
        """
        Perform attention head pruning.
        Args:
            kv_heads_to_prune_per_layer: List of KV heads to prune, format [(layer_index, [list of KV head indices to prune in that layer]), ...]
        """
        for layer_idx, kv_heads_to_prune in kv_heads_to_prune_per_layer:
            if not kv_heads_to_prune:
                continue

            # 1. Deduce Q heads to prune from KV heads to prune
            q_heads_to_prune = []
            for kv_head_idx in kv_heads_to_prune:
                start_q_idx = kv_head_idx * self.num_key_value_groups
                end_q_idx = (kv_head_idx + 1) * self.num_key_value_groups
                q_heads_to_prune.extend(list(range(start_q_idx, end_q_idx)))

            self._prune_layer_heads(layer_idx, sorted(q_heads_to_prune), sorted(kv_heads_to_prune))

    def _prune_layer_heads(self, layer_idx: int, q_heads_to_prune: List[int], kv_heads_to_prune: List[int]) -> None:
        """
        Prune attention heads in specified layer.
        Args:
            layer_idx: Layer index
            q_heads_to_prune: List of query (Q) head indices to prune
            kv_heads_to_prune: List of key/value (KV) head indices to prune
        """
        layer = self.model.layers[layer_idx].self_attn

        # 1. Determine remaining heads
        remaining_q_heads = sorted([h for h in range(self.original_num_q_heads) if h not in q_heads_to_prune])
        remaining_kv_heads = sorted([h for h in range(self.original_num_kv_heads) if h not in kv_heads_to_prune])

        # 2. Calculate new dimensions after pruning
        new_num_q_heads = len(remaining_q_heads)
        new_num_kv_heads = len(remaining_kv_heads)
        new_q_hidden_size = new_num_q_heads * self.head_dim
        new_kv_hidden_size = new_num_kv_heads * self.head_dim

        # 3. Prune weight matrices
        # q_proj
        new_q_weight = torch.cat([layer.q_proj.weight.data[h * self.head_dim:(h + 1) * self.head_dim]
                                  for h in remaining_q_heads], dim=0)
        new_q_proj = nn.Linear(self.original_hidden_size, new_q_hidden_size, bias=False,
                               device=layer.q_proj.weight.device)
        new_q_proj.weight = nn.Parameter(new_q_weight)
        layer.q_proj = new_q_proj

        # k_proj
        new_k_weight = torch.cat([layer.k_proj.weight.data[h * self.head_dim:(h + 1) * self.head_dim]
                                  for h in remaining_kv_heads], dim=0)
        new_k_proj = nn.Linear(self.original_hidden_size, new_kv_hidden_size, bias=False,
                               device=layer.k_proj.weight.device)
        new_k_proj.weight = nn.Parameter(new_k_weight)
        layer.k_proj = new_k_proj

        # v_proj
        new_v_weight = torch.cat([layer.v_proj.weight.data[h * self.head_dim:(h + 1) * self.head_dim]
                                  for h in remaining_kv_heads], dim=0)
        new_v_proj = nn.Linear(self.original_hidden_size, new_kv_hidden_size, bias=False,
                               device=layer.v_proj.weight.device)
        new_v_proj.weight = nn.Parameter(new_v_weight)
        layer.v_proj = new_v_proj

        # o_proj (input dimension changes)
        new_o_weight = torch.cat([layer.o_proj.weight.data[:, h * self.head_dim:(h + 1) * self.head_dim]
                                  for h in remaining_q_heads], dim=1)
        new_o_proj = nn.Linear(new_q_hidden_size, self.original_hidden_size, bias=False,
                               device=layer.o_proj.weight.device)
        new_o_proj.weight = nn.Parameter(new_o_weight)
        layer.o_proj = new_o_proj

        # 4. Update layer's attention head count configuration (most critical step)
        layer.num_heads = new_num_q_heads
        layer.num_attention_heads = new_num_q_heads
        layer.num_key_value_heads = new_num_kv_heads

        if new_num_kv_heads > 0 and new_num_q_heads % new_num_kv_heads == 0:
            layer.num_key_value_groups = new_num_q_heads // new_num_kv_heads
        else:
            # If MHA pruned to no KV heads, or cannot divide evenly (won't happen with our logic)
            layer.num_key_value_groups = 1 if new_num_kv_heads > 0 else 0

        layer.head_dim = self.head_dim

        # 5. Update top-level model config (very important, PEFT reads this)
        # Note: This modifies config for all layers, so should only be done after pruning all layers.
        # For simplicity, we update it here each time, assuming same pruning strategy across layers.
        self.config.num_attention_heads = new_num_q_heads
        self.config.num_key_value_heads = new_num_kv_heads

    def compute_head_importance(self, full_model: nn.Module, data_loader: DataLoader, device: str) -> List[torch.Tensor]:
        """
        Compute importance scores for each *KV* head across all layers.
        This method is based on the product of parameter gradients and parameter values (similar to SNIP).

        Args:
            data_loader: Data loader for gradient computation (a small batch is sufficient).
            device: Device where the model resides.

        Returns:
            A list where each element is a tensor of importance scores for KV heads in that layer.
        """
        print("    - Computing attention head importance...")

        # 1. Perform forward and backward pass to get gradients
        full_model.train()  # Ensure model is in training mode to compute gradients

        # Get a small batch of data from data loader
        try:
            batch = next(iter(data_loader))
        except StopIteration:
            raise ValueError("Data loader is empty. Cannot compute importance.")

        # Assume your dataloader outputs (input_ids, attention_mask, labels, opt_labels)
        # We only need input_ids and attention_mask
        input_ids, att_mask, _, _ = [x.to(device) for x in batch]

        # Zero out old gradients
        full_model.zero_grad()  # *** Modification: Zero gradients on full model ***

        # *** Modification: Use full_model for forward pass ***
        outputs = full_model(input_ids=input_ids, attention_mask=att_mask)
        logits = outputs.logits  # Now 'outputs' is of type CausalLMOutputWithPast, has .logits

        # Use same loss function as SNIP
        loss = lm_snip_loss(logits, input_ids)

        # Backward pass to compute gradients
        loss.backward()

        # 2. Compute importance for each KV head
        kv_head_importance_per_layer = []
        for layer in self.model.layers:
            attn = layer.self_attn

            # Initialize importance for all Q heads in current layer
            q_head_importance = torch.zeros(self.original_num_q_heads, device=device)

            # a) Importance from q_proj and o_proj (directly related to Q heads)
            # q_proj: weight shape [num_q_heads * head_dim, hidden_size]
            q_weight = attn.q_proj.weight
            q_grad = attn.q_proj.weight.grad
            q_saliency = (q_weight * q_grad).abs()
            for i in range(self.original_num_q_heads):
                start, end = i * self.head_dim, (i + 1) * self.head_dim
                q_head_importance[i] += q_saliency[start:end, :].sum()

            # o_proj: weight shape [hidden_size, num_q_heads * head_dim]
            o_weight = attn.o_proj.weight
            o_grad = attn.o_proj.weight.grad
            o_saliency = (o_weight * o_grad).abs()
            for i in range(self.original_num_q_heads):
                start, end = i * self.head_dim, (i + 1) * self.head_dim
                q_head_importance[i] += o_saliency[:, start:end].sum()

            # b) Importance from k_proj and v_proj (related to KV heads, then distributed to Q heads)
            # k_proj: weight shape [num_kv_heads * head_dim, hidden_size]
            k_weight = attn.k_proj.weight
            k_grad = attn.k_proj.weight.grad
            k_saliency = (k_weight * k_grad).abs()

            # v_proj: weight shape [num_kv_heads * head_dim, hidden_size]
            v_weight = attn.v_proj.weight
            v_grad = attn.v_proj.weight.grad
            v_saliency = (v_weight * v_grad).abs()

            for i in range(self.original_num_kv_heads):
                start, end = i * self.head_dim, (i + 1) * self.head_dim
                # Calculate importance of this KV head
                kv_head_saliency = k_saliency[start:end, :].sum() + v_saliency[start:end, :].sum()

                # Distribute this KV head's importance evenly to all Q heads it serves
                # This is a simplified approach but effective
                q_start_idx = i * self.num_key_value_groups
                q_end_idx = (i + 1) * self.num_key_value_groups
                q_head_importance[q_start_idx:q_end_idx] += kv_head_saliency / self.num_key_value_groups

            # c) Aggregate Q head importance to KV heads
            kv_head_importance = torch.zeros(self.original_num_kv_heads, device=device)
            for i in range(self.original_num_kv_heads):
                q_start_idx = i * self.num_key_value_groups
                q_end_idx = (i + 1) * self.num_key_value_groups
                # Sum importance of all Q heads in a group as the importance of that KV head
                kv_head_importance[i] = q_head_importance[q_start_idx:q_end_idx].sum()

            kv_head_importance_per_layer.append(kv_head_importance.cpu())

        # Clean up gradients to release memory
        full_model.zero_grad()  # *** Modification: Zero gradients on full model ***
        print("    - Attention head importance computation completed.")

        return kv_head_importance_per_layer

    def get_pruning_plan(self, importance_scores: List[torch.Tensor],
                         prune_ratio: float) -> List[Tuple[int, List[int]]]:
        """
        Generate pruning plan based on *KV* head importance scores.
        Args:
            importance_scores: Importance scores for KV heads in each layer
            prune_ratio: Pruning ratio (applied to KV heads)
        Returns:
            Pruning plan for KV heads
        """
        kv_heads_to_prune_per_layer = []
        # Calculate number of KV heads to prune based on total count, ensuring at least one head remains
        num_kv_heads_to_prune = min(int(self.original_num_kv_heads * prune_ratio), self.original_num_kv_heads - 1)

        if num_kv_heads_to_prune <= 0:
            return []

        for i, scores in enumerate(importance_scores):
            # argsort() returns [0, 1, 2, ...] on all-zero tensor
            prune_indices = scores.argsort()[:num_kv_heads_to_prune].tolist()
            kv_heads_to_prune_per_layer.append((i, prune_indices))

        return kv_heads_to_prune_per_layer


def apply_structured_pruning(stu_model, data_loader_for_pruning, device="cuda",
                             drop_layers_percent=0.2,
                             heads_prune_percent=0.25):
    """
    Perform structured pruning on LLaMA model
    - Transformer layer pruning
    - Attention head pruning
    """
    report_sparsity(stu_model, prefix="[Original Model Parameter Count] ")
    print(">>> Structured Pruning (Dynamic Layer Pruning) ...")

    # --------------------------
    # 1. Transformer Layer Pruning (Dynamic Version)
    # --------------------------
    original_num_layers = len(stu_model.model.layers)
    num_layers_to_drop = int(original_num_layers * drop_layers_percent)
    # print(f"[Step 1] Initial Transformer layers: {original_num_layers}")
    # print(f"         Planning to prune {num_layers_to_drop} layers ({drop_layers_percent:.0%})")

    if num_layers_to_drop > 0:
        # 1.1 Compute layer importance
        # print("    - Computing L1 norm importance of layers...")
        layer_importance = compute_layer_importance_by_norm(stu_model, norm_type="l1")
        # print(f"    - Layer importance scores: {[f'{x:.2f}' for x in layer_importance.tolist()]}")

        # 1.2 Determine layer indices to prune
        # argsort() default ascending, so least important first
        drop_idx = layer_importance.argsort()[:num_layers_to_drop].tolist()
        # print(f"    - Based on importance, decided to prune layer indices: {sorted(drop_idx)}")

        # 1.3 Perform pruning
        drop_transformer_layers(stu_model, drop_idx)
    else:
        print("Number of layers to prune is 0, skipping layer pruning.")

    after_num_layers = len(stu_model.model.layers)
    # print(f"[Step 1] Transformer layers after pruning: Remaining layers = {after_num_layers}\n")
    # report_sparsity(stu_model, prefix="[After Transformer Layer Pruning] ")

    # --------------------------
    # 2. Attention Head Pruning
    # --------------------------
    # print(f"[Step 2] Pruning attention heads: Pruning {heads_prune_percent * 100}% of KV head groups per layer")

    # Use LlamaAttentionHeadPruner to compute head importance
    pruner = LlamaAttentionHeadPruner(stu_model.model)  # Note passing model attribute here

    # *** Modification: Pass full stu_model ***
    head_importance = pruner.compute_head_importance(stu_model, data_loader_for_pruning, device)
    # Get pruning plan (now for KV heads)
    # Apply prune_ratio to KV heads
    kv_heads_to_prune = pruner.get_pruning_plan(head_importance, prune_ratio=heads_prune_percent)

    # Perform pruning
    total_pruned_kv_heads = 0
    if kv_heads_to_prune:
        for layer_idx, prune_kv_head_ids in kv_heads_to_prune:
            num_pruned_in_layer = len(prune_kv_head_ids)
            total_pruned_kv_heads += num_pruned_in_layer
            # Print pruned KV head groups
            # print(f"[Step 2] Layer {layer_idx}: Pruning KV head groups {prune_kv_head_ids} (Total pruned KV heads: {total_pruned_kv_heads})")

        # Perform attention head pruning
        pruner.prune_heads(kv_heads_to_prune)
    else:
        print("No KV heads were pruned.")

    # Print sparsity after pruning
    # report_sparsity(stu_model, prefix="[After Attention Head Pruning] ")
    #
    # print(">>> Structured pruning completed.\n")
    return stu_model


import gc


def find_threshold_from_list_of_tensors(scores_list, k):
    """
    Find the k-th largest value without merging the list of tensors.
    This avoids creating huge intermediate tensors.
    """
    print("        - Converting score list to Numpy arrays...")
    # Convert tensor list to list of numpy arrays, then immediately release tensor list
    np_scores_list = [t.numpy() for t in scores_list]
    del scores_list
    gc.collect()

    print("        - Merging Numpy arrays...")

    all_scores_np = np.concatenate(np_scores_list, axis=None)
    del np_scores_list
    gc.collect()

    print(f"        - Using np.partition to find the {k}th largest value...")
    # np.partition(a, k) places the k-th smallest element in correct position, smaller elements to the left, larger to the right
    if k >= all_scores_np.size:
        # If number to keep is greater than or equal to total, no pruning, threshold is 0
        return 0.0

    k_smallest_index = all_scores_np.size - k
    # Use argpartition to find index of k-th largest element, then get its value
    threshold_val = np.partition(all_scores_np, k_smallest_index)[k_smallest_index]

    del all_scores_np
    gc.collect()

    return threshold_val


def snip_prune_with_mask_ultimate(model, data_loader, device, sparsity=0.3, chunk_size=1000000):
    """
    Ultimate version of SNIP pruning function: Minimize CPU memory peak by avoiding torch.cat.
    """
    print("    - Starting ultimate SNIP pruning (without torch.cat)...")
    model.train()

    # 1. Compute gradients (same as before)
    print("      - Step 1/3: Computing gradients...")
    model.to(device)
    batch = next(iter(data_loader))
    input_ids, att_mask, _, _ = [x.to(device) for x in batch]
    model.zero_grad()
    output = model(input_ids=input_ids, attention_mask=att_mask)
    logits = output.logits
    task_loss = lm_snip_loss(logits, input_ids)
    task_loss.backward()
    print("      - Gradient computation completed.")

    # 2. Compute scores in chunks and collect in CPU list, without merging
    print("      - Step 2/3: Computing importance scores in chunks...")
    all_scores_cpu_list = []
    total_params = 0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                total_params += p.numel()
                p_flat = p.data.flatten()
                g_flat = p.grad.data.flatten()
                for i in range(0, p.numel(), chunk_size):
                    end = i + chunk_size
                    p_chunk = p_flat[i:end].float()
                    g_chunk = g_flat[i:end].float()
                    saliency_chunk = (p_chunk * g_chunk).abs().cpu()
                    all_scores_cpu_list.append(saliency_chunk)

    print(f"      - Score computation completed, total parameters: {total_params}. Finding threshold...")

    # Calculate number of parameters to keep
    k = max(1, int(total_params * (1 - sparsity)))

    # *** Key modification: Use new function to find threshold, avoiding torch.cat ***
    threshold_np = find_threshold_from_list_of_tensors(all_scores_cpu_list, k)
    threshold = torch.tensor(threshold_np, device=device)

    # find_threshold_from_list_of_tensors has already cleaned up the list internally
    gc.collect()
    print(f"      - Pruning threshold determined: {threshold.item()}")

    # 3. Apply pruning mask in chunks (same as before)
    print("      - Step 3/3: Applying pruning mask...")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                p_flat = p.data.flatten()
                g_flat = p.grad.data.flatten()
                for i in range(0, p.numel(), chunk_size):
                    end = i + chunk_size
                    p_chunk = p_flat[i:end].float()
                    g_chunk = g_flat[i:end].float()
                    saliency_chunk = (p_chunk * g_chunk).abs()
                    mask_chunk = saliency_chunk >= threshold
                    p_flat[i:end].mul_(mask_chunk.to(p_flat.dtype))
                # Write modified flattened tensor back to original parameter (not needed as mul_ is in-place)

    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    print("    - SNIP pruning completed.")


def lm_snip_loss(logits, input_ids):
    """
    Compute cross-entropy loss for autoregressive language model, used for gradient computation in SNIP pruning.

    Parameters:
        logits: Log probability distribution predicted by model, shape [batch_size, seq_len, vocab_size].
        input_ids: Input token IDs, shape [batch_size, seq_len].

    Returns:
        Cross-entropy loss (scalar).
    """
    # Shift operation: model predicts token at position t+1 using position t
    shift_logits = logits[..., :-1, :].contiguous()  # Remove prediction of last timestep
    shift_labels = input_ids[..., 1:].contiguous()  # Remove first token as target

    # Compute cross-entropy loss
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),  # [batch_size*(seq_len-1), vocab_size]
        shift_labels.view(-1),  # [batch_size*(seq_len-1)]
        reduction="mean"
    )


def gradual_magnitude_prune(model, current_epoch, total_epochs,
                             final_sparsity=0.5):
    ratio = final_sparsity * (current_epoch / total_epochs)
    for p in model.parameters():
        if p.requires_grad and p.ndim > 1:
            thresh = torch.quantile(p.abs().flatten(), ratio)
            p.data.mul_((p.abs() >= thresh).float())


# ====================== 3. Sparsity Report ======================================
def report_sparsity(model: nn.Module, prefix=""):
    """
    Statistically report sparsity on GPU in one pass, minimal CPU/GPU data transfer.
    If GPU memory is tight, can be changed to per-layer traversal or .half() before statistics.
    """
    total = nz = 0
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 0:      # scalar (e.g., Logit scale), can be ignored
                continue
            total += p.numel()
            nz += p.detach().cpu().count_nonzero().item()
    sparsity = 1 - nz / total
    print(f"{prefix}Total params: {total:,} | Non-zero: {nz:,} | Sparsity: {sparsity:.2%}")
    return total, nz