import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------- Project Dependencies ----------
from loss import EnhancedVACCLoss
from Embedding import PlanGAT, TimeLSTM
from CombinedDataset import CombinedDataset
from Combine_model import CombinedModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from sql_dataset import TrainDataset
from model import Llama4SQA
from distillation import dynamic_loss, compute_jsd_sigmoid
from torch_geometric.data import Data, Batch
from subdateset import CustomSubset
from model_prompt import Llama4SQA_prompt
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple
import hashlib
import time
from peft import PeftModel
import os
import pandas as pd
import itertools


def measure_gpu_memory(model, device, note=""):
    """
    Measure and print GPU memory usage of the model.
    Args:
        model: The model to measure.
        device: Device where the model resides.
        note (str): Optional note to print with the message.
    """
    model.to(device)
    torch.cuda.empty_cache()

    allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 2)

    print(f"{note} GPU Memory Usage -> Allocated: {allocated_memory:.2f} MiB | Reserved: {reserved_memory:.2f} MiB")
    return allocated_memory


def run_pruning_experiment(args, snip_sparsity, drop_layers_percent, heads_prune_percent):
    """
    Run a complete pruning and evaluation experiment.
    """
    device = args.device
    print("\n" + "=" * 80)
    print(
        f"== Starting New Experiment: SNIP Sparsity={snip_sparsity:.2f}, Layer Prune={drop_layers_percent:.2%}, Head Prune={heads_prune_percent:.2%} ==")
    print("=" * 80)

    # <<< MODIFICATION START: Load and merge pre-trained LoRA weights >>>
    # --------------------------
    # 1. Load base model and merge pre-trained LoRA adapter
    # --------------------------
    model_path = args.pre_path + '/models/llama-1B/'
    tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    tokenizer.pad_token = tokenizer.eos_token

    # 1.1 Load base model (without LoRA)
    base_model = AutoModelForCausalLM.from_pretrained(model_path)

    # 1.2 Load pre-trained LoRA adapter
    lora_adapter_path = args.lora2edge
    model_with_lora = PeftModel.from_pretrained(base_model, lora_adapter_path)

    # 1.3 Merge LoRA weights into base model and unload adapter
    student_model = model_with_lora.merge_and_unload()
    for param in student_model.parameters():
        param.requires_grad = True

    student_model.to(device)
    student_model.train()  # Ensure gradients are computed (needed for pruning)

    # --------------------------
    # 2. Unstructured pruning (SNIP) with mask
    # --------------------------
    print("\n--- Performing Unstructured Pruning (SNIP) ---")
    train_data_prue = TrainDataset(args, tokenizer, 'train')
    dataloader_train_prue = DataLoader(train_data_prue, batch_size=args.batch_size, shuffle=True)

    if snip_sparsity > 0:
        snip_prune_with_mask_ultimate(
            model=student_model,
            data_loader=dataloader_train_prue,
            device=device,
            sparsity=snip_sparsity
        )
    # Validate pruning effect
    report_sparsity(student_model, prefix="[After SNIP Pruning] ")

    # --------------------------
    # 3. Structured pruning
    # --------------------------
    print("\n--- Performing Structured Pruning ---")
    student_model = apply_structured_pruning(
        student_model,
        dataloader_train_prue,
        device,
        drop_layers_percent=drop_layers_percent,
        heads_prune_percent=heads_prune_percent
    )

    # --- Get final parameter count ---
    final_total_params, final_nonzero_params = report_sparsity(student_model, prefix="[Final Model] ")

    # Note: Classifier head uses original hidden size
    original_hidden_size = student_model.config.hidden_size
    student_model = Llama4SQA_prompt(student_model, num_labels=9, hidden_size=original_hidden_size).to(device)

    # Load classifier weights
    classifiers_path = args.cls2edge
    if os.path.exists(classifiers_path):
        classifiers_state = torch.load(classifiers_path, map_location=device, weights_only=True)
        student_model.classifiers.load_state_dict(classifiers_state)
        print("Classifier weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Classifier weights not found at {classifiers_path}")

    # Set model to evaluation mode
    student_model.eval()

    # 5. Load test dataset and dataloader
    test_data_student = TrainDataset(args, tokenizer, 'test')
    dataloader_test_stu = DataLoader(test_data_student, batch_size=args.batch_size, shuffle=False)
    test_len = len(dataloader_test_stu)

    # 6. Inference loop
    with torch.no_grad():
        all_pred_opt = []
        all_opt_labels = []
        # <<< MODIFICATION START: Initialize inference timer >>>
        total_inference_time = 0.0
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # <<< MODIFICATION END >>>

        for batch_idx, batch in enumerate(tqdm(dataloader_test_stu, leave=True)):
            input_ids, att_mask, labels, opt_labels = batch
            input_ids = input_ids.to(device)
            att_mask = att_mask.to(device)
            labels = labels.to(device)
            opt_labels = opt_labels.to(device)

            # <<< MODIFICATION START: Measure inference time >>>
            starter.record()
            logits = student_model(input_ids=input_ids, attention_mask=att_mask)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # in milliseconds
            total_inference_time += curr_time
            # <<< MODIFICATION END >>>

            probs = torch.sigmoid(logits)
            pred_opt = probs

            all_pred_opt.append(pred_opt)
            all_opt_labels.append(opt_labels)

        # Concatenate all batches
        all_pred_opt = torch.cat(all_pred_opt, dim=0)
        all_opt_labels = torch.cat(all_opt_labels, dim=0)
        avg_inference_time_ms = total_inference_time / test_len if test_len > 0 else 0

        # Compute regression metrics
        mse = torch.mean((all_pred_opt - all_opt_labels) ** 2).item()
        mae = torch.mean(torch.abs(all_pred_opt - all_opt_labels)).item()

        # Compute V-ACC
        all_multilabel_pred = (all_pred_opt > 0.5).float()
        all_multilabel_true = (all_opt_labels > 0.1).float()
        right_label_all = torch.where(all_multilabel_true == all_multilabel_pred, 1, 0).sum()
        vacc = right_label_all / (test_len * 9)  # assuming 9 labels per sample

        # Compute additional metrics using calculate_metrics
        metrics = calculate_metrics(all_multilabel_pred, all_opt_labels, device=device)
        report_dict = metrics.pop('report_dict')
        precision_1 = report_dict.get('1.0', {}).get('precision', 0.0)

        # --- Get GPU memory usage ---
        gpu_mem = measure_gpu_memory(student_model, device, note="[Pruned Model]")

        # --- Assemble results ---
        results = {
            'SNIP Sparsity': snip_sparsity,
            'Layer Prune %': drop_layers_percent,
            'Head Prune %': heads_prune_percent,
            'V-ACC': vacc.item(),
            'F1 Macro': metrics['F1_macro'],
            'AUC Micro': metrics['AUC_micro'],
            'Precision': precision_1,
            'Total Params': final_total_params,
            'Non-zero Params': final_nonzero_params,
            'GPU (MiB)': gpu_mem,
            'Inference Time (ms/batch)': avg_inference_time_ms,
        }

        # Print results
        print(f'Test MSE: {mse:.4f}, MAE: {mae:.4f}, V-ACC: {vacc:.4f}')
        print("\nAdditional Metrics:")
        print(f"F1 Micro: {metrics['F1_micro']:.4f}")
        print(f"F1 Macro: {metrics['F1_macro']:.4f}")
        print(f"AUC Micro: {metrics['AUC_micro']:.4f}")
        print(f"Precision_1: {results['Precision']}")
        print(f"Average Inference Time: {avg_inference_time_ms:.2f} ms/batch")

    print('---------------- Experiment Completed ----------------')

    # Cleanup
    del student_model, base_model, model_with_lora, train_data_prue, dataloader_train_prue, test_data_student, dataloader_test_stu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def prun_experiment_loop_edge(args):
    # Define experiment parameter grid
    experiment_grid = {
        'snip_sparsity': [0, 0.15, 0.3, 0.45],
        'drop_layers_percent': [0, 0.15, 0.3, 0.45],
        'heads_prune_percent': [0, 0.15, 0.3, 0.45],
    }

    all_results = []

    keys, values = zip(*experiment_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Ensure baseline (no pruning) runs first
    baseline_params = {key: 0.0 for key in keys}
    if baseline_params in param_combinations:
        param_combinations.remove(baseline_params)
    param_combinations.insert(0, baseline_params)

    for params in tqdm(param_combinations, desc="All Experiments"):
        try:
            result = run_pruning_experiment(args, **params)
            all_results.append(result)
        except Exception as e:
            print(f"\n!!!!!! Experiment Failed: {params}, Error: {e} !!!!!!\n")
            import traceback
            traceback.print_exc()
            error_result = {**params, 'V-ACC': 'Error', 'F1 Macro': str(e)}
            all_results.append(error_result)

    if not all_results:
        print("No experiment results to export.")
        return

    results_df = pd.DataFrame(all_results)

    # Format and compute derived columns
    baseline_total_params = results_df.loc[0, 'Total Params']
    baseline_nonzero_params = results_df.loc[0, 'Non-zero Params']

    results_df['Total Param Reduction %'] = (1 - results_df['Total Params'] / baseline_total_params) * 100
    results_df['Non-zero Param Reduction %'] = (1 - results_df['Non-zero Params'] / baseline_nonzero_params) * 100

    # Reorder columns for readability
    display_cols = [
        'SNIP Sparsity', 'Layer Prune %', 'Head Prune %',
        'V-ACC', 'F1 Macro', 'AUC Micro', 'Precision',
        'Total Params', 'Total Param Reduction %',
        'Non-zero Params', 'Non-zero Param Reduction %',
        'GPU (MiB)', 'Inference Time (ms/batch)'
    ]
    results_df = results_df[display_cols]

    print("\n\n" + "=" * 40 + " Final Experiment Results " + "=" * 40)
    print(results_df.to_string())

    # Export to CSV
    output_path = args.pre_path + f"/combined_model/pruning_experiment_results_{args.dataset}.csv"
    results_df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\nResults saved to: {output_path}")


def calculate_metrics(all_pred_labels, all_opt_labels, device='cuda:0'):
    """Calculate evaluation metrics including F1, AUC, and Subset Accuracy."""
    all_pred_labels = all_pred_labels.to(device)
    all_opt_labels = all_opt_labels.to(device)

    y_pred = all_pred_labels.cpu().numpy()
    y_true = (all_opt_labels > 0.1).float().cpu().numpy()

    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()

    f1_micro = f1_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
    f1_macro = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)

    try:
        auc_micro = roc_auc_score(y_true_flat, y_pred.flatten(), average='micro')
    except ValueError:
        auc_micro = 0.0

    subset_accuracy = torch.all(all_pred_labels == torch.from_numpy(y_true).to(device), dim=1).float().mean()

    report_dict = classification_report(y_true_flat, y_pred_flat, output_dict=True, zero_division=0)

    return {
        'F1_micro': f1_micro,
        'F1_macro': f1_macro,
        'AUC_micro': auc_micro,
        'Subset_Accuracy': subset_accuracy.item(),
        'report_dict': report_dict
    }


# ====================== 1. Structured Pruning Tools ================================

def compute_layer_importance_by_norm(model: nn.Module, norm_type: str = "l1") -> torch.Tensor:
    """
    Compute layer importance based on parameter norms.
    Args:
        model: The model to evaluate.
        norm_type: 'l1' or 'l2'.
    Returns:
        Tensor of importance scores for each transformer layer.
    """
    layer_importances = []
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
# 2. Transformer Layer Pruning
# --------------------------

def drop_transformer_layers(hf_llama, drop_idx):
    """
    Remove specified layers and update model configuration.
    """
    kept_layers = [layer for i, layer in enumerate(hf_llama.model.layers) if i not in drop_idx]
    hf_llama.model.layers = nn.ModuleList(kept_layers)
    hf_llama.model.config.num_hidden_layers = len(kept_layers)
    hf_llama.load_state_dict(hf_llama.state_dict(), strict=False)
    hf_llama.num_parameters = sum(p.numel() for p in hf_llama.parameters())
    print(f"Model parameter count after pruning: {hf_llama.num_parameters}")


# --------------------------
# 3. Llama Attention Head Pruning (GQA-Aware)
# --------------------------

class LlamaAttentionHeadPruner:
    """
    Tool for pruning attention heads in Llama models (supports GQA/MQA, group-wise pruning).
    """

    def __init__(self, model):
        self.model = model
        self.config = model.config
        self.original_hidden_size = self.config.hidden_size
        self.original_num_q_heads = self.config.num_attention_heads
        self.original_num_kv_heads = getattr(self.config, 'num_key_value_heads', self.original_num_q_heads)
        self.head_dim = self.original_hidden_size // self.original_num_q_heads

        if self.original_num_q_heads % self.original_num_kv_heads != 0:
            raise ValueError("Q heads must be a multiple of KV heads for GQA.")
        self.num_key_value_groups = self.original_num_q_heads // self.original_num_kv_heads

    def prune_heads(self, kv_heads_to_prune_per_layer: List[Tuple[int, List[int]]]) -> None:
        """
        Prune attention heads.
        Args:
            kv_heads_to_prune_per_layer: List of (layer_idx, [kv_head_indices]) to prune.
        """
        for layer_idx, kv_heads_to_prune in kv_heads_to_prune_per_layer:
            if not kv_heads_to_prune:
                continue
            q_heads_to_prune = []
            for kv_head_idx in kv_heads_to_prune:
                start_q_idx = kv_head_idx * self.num_key_value_groups
                end_q_idx = (kv_head_idx + 1) * self.num_key_value_groups
                q_heads_to_prune.extend(range(start_q_idx, end_q_idx))
            self._prune_layer_heads(layer_idx, sorted(q_heads_to_prune), sorted(kv_heads_to_prune))

    def _prune_layer_heads(self, layer_idx: int, q_heads_to_prune: List[int], kv_heads_to_prune: List[int]) -> None:
        layer = self.model.layers[layer_idx].self_attn
        remaining_q_heads = sorted([h for h in range(self.original_num_q_heads) if h not in q_heads_to_prune])
        remaining_kv_heads = sorted([h for h in range(self.original_num_kv_heads) if h not in kv_heads_to_prune])

        new_num_q_heads = len(remaining_q_heads)
        new_num_kv_heads = len(remaining_kv_heads)
        new_q_hidden_size = new_num_q_heads * self.head_dim
        new_kv_hidden_size = new_num_kv_heads * self.head_dim

        # Prune q_proj
        new_q_weight = torch.cat([layer.q_proj.weight.data[h * self.head_dim:(h + 1) * self.head_dim]
                                  for h in remaining_q_heads], dim=0)
        layer.q_proj = nn.Linear(self.original_hidden_size, new_q_hidden_size, bias=False,
                                 device=layer.q_proj.weight.device)
        layer.q_proj.weight = nn.Parameter(new_q_weight)

        # Prune k_proj
        new_k_weight = torch.cat([layer.k_proj.weight.data[h * self.head_dim:(h + 1) * self.head_dim]
                                  for h in remaining_kv_heads], dim=0)
        layer.k_proj = nn.Linear(self.original_hidden_size, new_kv_hidden_size, bias=False,
                                 device=layer.k_proj.weight.device)
        layer.k_proj.weight = nn.Parameter(new_k_weight)

        # Prune v_proj
        new_v_weight = torch.cat([layer.v_proj.weight.data[h * self.head_dim:(h + 1) * self.head_dim]
                                  for h in remaining_kv_heads], dim=0)
        layer.v_proj = nn.Linear(self.original_hidden_size, new_kv_hidden_size, bias=False,
                                 device=layer.v_proj.weight.device)
        layer.v_proj.weight = nn.Parameter(new_v_weight)

        # Prune o_proj
        new_o_weight = torch.cat([layer.o_proj.weight.data[:, h * self.head_dim:(h + 1) * self.head_dim]
                                  for h in remaining_q_heads], dim=1)
        layer.o_proj = nn.Linear(new_q_hidden_size, self.original_hidden_size, bias=False,
                                 device=layer.o_proj.weight.device)
        layer.o_proj.weight = nn.Parameter(new_o_weight)

        # Update layer config
        layer.num_heads = new_num_q_heads
        layer.num_attention_heads = new_num_q_heads
        layer.num_key_value_heads = new_num_kv_heads
        layer.num_key_value_groups = new_num_q_heads // new_num_kv_heads if new_num_kv_heads > 0 else 0
        layer.head_dim = self.head_dim

        # Update global config
        self.config.num_attention_heads = new_num_q_heads
        self.config.num_key_value_heads = new_num_kv_heads

    def compute_head_importance(self, full_model: nn.Module, data_loader: DataLoader, device: str) -> List[
        torch.Tensor]:
        """
        Compute importance of each KV head based on gradient × parameter.
        """
        print("    - Computing head importance...")
        full_model.train()
        try:
            batch = next(iter(data_loader))
        except StopIteration:
            raise ValueError("Data loader is empty.")
        input_ids, att_mask, _, _ = [x.to(device) for x in batch]
        full_model.zero_grad()
        outputs = full_model(input_ids=input_ids, attention_mask=att_mask)
        logits = outputs.logits
        loss = lm_snip_loss(logits, input_ids)
        loss.backward()

        kv_head_importance_per_layer = []
        for layer in self.model.layers:
            attn = layer.self_attn
            q_head_importance = torch.zeros(self.original_num_q_heads, device=device)

            # Q and O projections
            q_weight = attn.q_proj.weight
            q_grad = attn.q_proj.weight.grad
            q_saliency = (q_weight * q_grad).abs()
            for i in range(self.original_num_q_heads):
                start, end = i * self.head_dim, (i + 1) * self.head_dim
                q_head_importance[i] += q_saliency[start:end, :].sum()

            o_weight = attn.o_proj.weight
            o_grad = attn.o_proj.weight.grad
            o_saliency = (o_weight * o_grad).abs()
            for i in range(self.original_num_q_heads):
                start, end = i * self.head_dim, (i + 1) * self.head_dim
                q_head_importance[i] += o_saliency[:, start:end].sum()

            # K and V projections
            k_weight = attn.k_proj.weight
            k_grad = attn.k_proj.weight.grad
            k_saliency = (k_weight * k_grad).abs()

            v_weight = attn.v_proj.weight
            v_grad = attn.v_proj.weight.grad
            v_saliency = (v_weight * v_grad).abs()

            kv_head_importance = torch.zeros(self.original_num_kv_heads, device=device)
            for i in range(self.original_num_kv_heads):
                start, end = i * self.head_dim, (i + 1) * self.head_dim
                kv_head_saliency = k_saliency[start:end, :].sum() + v_saliency[start:end, :].sum()
                q_start_idx = i * self.num_key_value_groups
                q_end_idx = (i + 1) * self.num_key_value_groups
                q_head_importance[q_start_idx:q_end_idx] += kv_head_saliency / self.num_key_value_groups

            for i in range(self.original_num_kv_heads):
                q_start_idx = i * self.num_key_value_groups
                q_end_idx = (i + 1) * self.num_key_value_groups
                kv_head_importance[i] = q_head_importance[q_start_idx:q_end_idx].sum()

            kv_head_importance_per_layer.append(kv_head_importance.cpu())

        full_model.zero_grad()
        print("    - Head importance computation completed.")
        return kv_head_importance_per_layer

    def get_pruning_plan(self, importance_scores: List[torch.Tensor], prune_ratio: float) -> List[
        Tuple[int, List[int]]]:
        num_kv_heads_to_prune = min(int(self.original_num_kv_heads * prune_ratio), self.original_num_kv_heads - 1)
        if num_kv_heads_to_prune <= 0:
            return []
        kv_heads_to_prune_per_layer = []
        for i, scores in enumerate(importance_scores):
            prune_indices = scores.argsort()[:num_kv_heads_to_prune].tolist()
            kv_heads_to_prune_per_layer.append((i, prune_indices))
        return kv_heads_to_prune_per_layer


def apply_structured_pruning(stu_model, data_loader_for_pruning, device="cuda",
                             drop_layers_percent=0.2, heads_prune_percent=0.25):
    """
    Apply structured pruning to LLaMA model:
    - Transformer layer pruning
    - Attention head pruning
    """
    report_sparsity(stu_model, prefix="[Original Model] ")
    print(">>> Structured Pruning (Dynamic Layer Pruning) ...")

    # 1. Layer pruning
    original_num_layers = len(stu_model.model.layers)
    num_layers_to_drop = int(original_num_layers * drop_layers_percent)
    if num_layers_to_drop > 0:
        layer_importance = compute_layer_importance_by_norm(stu_model, norm_type="l1")
        drop_idx = layer_importance.argsort()[:num_layers_to_drop].tolist()
        drop_transformer_layers(stu_model, drop_idx)
    else:
        print("No layers to prune, skipping layer pruning.")

    after_num_layers = len(stu_model.model.layers)

    # 2. Head pruning
    pruner = LlamaAttentionHeadPruner(stu_model.model)
    head_importance = pruner.compute_head_importance(stu_model, data_loader_for_pruning, device)
    kv_heads_to_prune = pruner.get_pruning_plan(head_importance, prune_ratio=heads_prune_percent)

    total_pruned_kv_heads = 0
    if kv_heads_to_prune:
        for layer_idx, prune_kv_head_ids in kv_heads_to_prune:
            total_pruned_kv_heads += len(prune_kv_head_ids)
        pruner.prune_heads(kv_heads_to_prune)
    else:
        print("No KV heads to prune.")

    return stu_model


# ====================== 2. Unstructured Pruning Tools ================================

import gc


def find_threshold_from_list_of_tensors(scores_list, k):
    """
    Find the k-th largest value without concatenating tensors to save memory.
    """
    print("        - Converting scores to numpy arrays...")
    np_scores_list = [t.numpy() for t in scores_list]
    del scores_list
    gc.collect()

    print("        - Concatenating numpy arrays...")
    all_scores_np = np.concatenate(np_scores_list, axis=None)
    del np_scores_list
    gc.collect()

    print(f"        - Finding the {k}-th largest value using np.partition...")
    if k >= all_scores_np.size:
        return 0.0
    k_smallest_index = all_scores_np.size - k
    threshold_val = np.partition(all_scores_np, k_smallest_index)[k_smallest_index]

    del all_scores_np
    gc.collect()
    return threshold_val


def snip_prune_with_mask_ultimate(model, data_loader, device, sparsity=0.3, chunk_size=1000000):
    """
    Ultimate SNIP pruning function that avoids torch.cat to minimize CPU memory usage.
    """
    print("    - Starting Ultimate SNIP Pruning (no torch.cat)...")
    model.train()
    print("      - Step 1/3: Computing gradients...")
    model.to(device)
    batch = next(iter(data_loader))
    input_ids, att_mask, _, _ = [x.to(device) for x in batch]
    model.zero_grad()
    output = model(input_ids=input_ids, attention_mask=att_mask)
    logits = output.logits
    task_loss = lm_snip_loss(logits, input_ids)
    task_loss.backward()
    print("      - Gradients computed.")

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

    k = max(1, int(total_params * (1 - sparsity)))
    threshold_np = find_threshold_from_list_of_tensors(all_scores_cpu_list, k)
    threshold = torch.tensor(threshold_np, device=device)
    gc.collect()
    print(f"      - Pruning threshold determined: {threshold.item()}")

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
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    print("    - SNIP pruning completed.")


def lm_snip_loss(logits, input_ids):
    """
    Compute cross-entropy loss for autoregressive language modeling (used in SNIP).
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean"
    )


def gradual_magnitude_prune(model, current_epoch, total_epochs, final_sparsity=0.5):
    ratio = final_sparsity * (current_epoch / total_epochs)
    for p in model.parameters():
        if p.requires_grad and p.ndim > 1:
            thresh = torch.quantile(p.abs().flatten(), ratio)
            p.data.mul_((p.abs() >= thresh).float())


# ====================== 3. Sparsity Reporting ======================================
def report_sparsity(model: nn.Module, prefix=""):
    """
    Report sparsity with minimal CPU/GPU data transfer.
    """
    total = nz = 0
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 0:
                continue
            total += p.numel()
            nz += p.detach().cpu().count_nonzero().item()
    sparsity = 1 - nz / total
    print(f"{prefix}Total params: {total:,} | Non-zero: {nz:,} | Sparsity: {sparsity:.2%}")
    return total, nz