import os
import torch.nn as nn
from torch.utils.data import DataLoader,Subset
from transformers import LlamaForCausalLM, AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from sql_dataset import TrainDataset
from prompt import construct_prompt_lora
from tqdm import tqdm
from model import Llama4SQA
from sklearn.metrics import f1_score, jaccard_score
import torch.nn.functional as F
import numpy as np
from torch.optim import lr_scheduler
from loss import EnhancedVACCLoss

# lora fine-tune
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import classification_report
from model_prompt import Llama4SQA_prompt
import time
def train_llm(args):
    print('------------------------Lora Fine-Tune Start--------------------------')
# -----------------------------------------------------------------------------------------------------------------------#
    # LLAMA3.1-1B
    model_path = args.pre_path + '/models/llama-1B/'
# -----------------------------------------------------------------------------------------------------------------------#
    tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    tokenizer.pad_token = tokenizer.eos_token
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    device = args.device
    llama = AutoModelForCausalLM.from_pretrained(model_path)
    llama_lora = get_peft_model(llama, lora_config)
    model = Llama4SQA_prompt(llama_lora, num_labels=9).to(device)
# -----------------------------------------------------------------------------------------------------------------------#
    train_data = TrainDataset(args, tokenizer, 'train')
    val_data = TrainDataset(args, tokenizer, 'val')
    test_data = TrainDataset(args, tokenizer, 'test')
# -----------------------------------------------------------------------------------------------------------------------#
    dataloader_train = DataLoader(train_data, batch_size=1, shuffle=False)
    dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False)
    dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False)
# -----------------------------------------------------------------------------------------------------------------------#
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# -----------------------------------------------------------------------------------------------------------------------#
    opt_threshold = 0.1
# -----------------------------------------------------------------------------------------------------------------------#
    epsilon = 0.1
    loss_fn = EnhancedVACCLoss(
        epsilon=epsilon,
        alpha=1,
        beta=0,
        gamma=0
    ).to(device)
# -----------------------------------------------------BASELINE------------------------------------------------------------------#
    test_len = len(test_data)
    val_len = len(val_data)
    model.eval()
    with torch.no_grad():
        for test_idx in range(1):
            base_all_pred_opt = []
            base_all_opt_labels = []
            for batch_idx, batch in enumerate(tqdm(dataloader_test, leave=True)):
                input_ids, att_mask, labels, opt_labels = batch
                input_ids, att_mask, labels, opt_labels = input_ids.to(device), att_mask.to(device), labels.to(device), opt_labels.to(device)
                logits = model(input_ids=input_ids, attention_mask=att_mask)
                probs = torch.sigmoid(logits)
                # probs = logits
                pred_opt = probs
                base_all_pred_opt.append(pred_opt)
                base_all_opt_labels.append(opt_labels)

            all_pred_opt = torch.cat(base_all_pred_opt, dim=0)
            all_opt_labels = torch.cat(base_all_opt_labels, dim=0)
# ------------------------------------------------------------------------------------------------------------------------------#

            mse = torch.mean((all_pred_opt - all_opt_labels) ** 2).item()
            mae = torch.mean(torch.abs(all_pred_opt - all_opt_labels)).item()

            all_multilabel_pred = (all_pred_opt > 0.5).float()
            all_multilabel_true = (all_opt_labels > opt_threshold).float()
            right_label_all = torch.where(all_multilabel_true == all_multilabel_pred, 1, 0).sum()
            vacc = right_label_all / test_len / 9
            print(f'Baseline---Test MSE: {mse}, MAE: {mae}, V-ACC: {vacc}')
            metrics = calculate_metrics(all_multilabel_pred, all_opt_labels)
            print("\nNewly added evaluation metrics:")
            print(f"F1 Micro: {metrics['F1_micro']:.4f}")
            print(f"F1 Macro: {metrics['F1_macro']:.4f}")
            print(f"AUC Micro: {metrics['AUC_micro']:.4f}")
            print(f"AUC Macro: {metrics['AUC_macro']:.4f}")
            print(f"Subset Accuracy: {metrics['Subset_Accuracy']:.4f}")
# -----------------------------------------------------TRAIN------------------------------------------------------------------#
    for epoch in range(args.epochs):

        epoch_start_time = time.time()

        current_lr = optimizer.param_groups[0]['lr']
        print(f'<<------------------- epoch {epoch}/{args.epochs} , lr: {current_lr}------------------>>')
        model.train()
        total_loss = 0
        accumulated_loss = 0
        accumulation_steps = 1
        for batch_idx, batch in enumerate(tqdm(dataloader_train, leave=True)):

            input_ids, att_mask, labels, opt_labels = batch
            input_ids, att_mask, labels, opt_labels = input_ids.to(device), att_mask.to(device), labels.to(device), opt_labels.to(device)
            outputs = model(input_ids=input_ids, attention_mask=att_mask)

            loss, binary_loss, regression_loss, threshold_loss = loss_fn(
                outputs, labels, opt_labels
            )

            accumulated_loss += loss
            total_loss += loss.item()
# ----------------------------------------------------------BACKWARD-------------------------------------------------------------#

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader_train):

                avg_loss = accumulated_loss / min(accumulation_steps, (batch_idx + 1) % accumulation_steps or accumulation_steps)


                avg_loss.backward()


                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)


                optimizer.step()


                accumulated_loss = 0
                optimizer.zero_grad()
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        avg_epoch_loss = total_loss / len(dataloader_train)
        print(f'Epoch {epoch} Completed, Average loss: {avg_epoch_loss:.4f}，time-consuming：{epoch_duration}')
# -----------------------------------------------------------------------------------------------------------------------#
# ----------------------------------------------------TEST-VAL-VACC-------------------------------------------------------------------#
        model.eval()
        with torch.no_grad():
            all_pred_opt = []
            all_opt_labels = []
            eval_start_time = time.time()
            for batch_idx, batch in enumerate(tqdm(dataloader_test, leave=True) ):

                input_ids, att_mask, labels, opt_labels = batch
                input_ids, att_mask, labels, opt_labels = input_ids.to(device), att_mask.to(device), labels.to(device), opt_labels.to(device)
                logits = model(input_ids=input_ids, attention_mask=att_mask)
                probs = torch.sigmoid(logits)
                # probs = logits
                pred_opt = probs
                all_pred_opt.append(pred_opt)
                all_opt_labels.append(opt_labels)

            all_pred_opt = torch.cat(all_pred_opt, dim=0)
            all_opt_labels = torch.cat(all_opt_labels, dim=0)

            mse = torch.mean((all_pred_opt - all_opt_labels) ** 2).item()
            mae = torch.mean(torch.abs(all_pred_opt - all_opt_labels)).item()

            all_multilabel_pred = (all_pred_opt > 0.5).float()
            all_multilabel_true = (all_opt_labels > opt_threshold).float()
            right_label_all = torch.where(all_multilabel_true == all_multilabel_pred, 1, 0).sum()
            vacc = right_label_all / test_len / 9
            metrics = calculate_metrics(all_multilabel_pred, all_opt_labels)
            print("\nNewly added evaluation metrics:")
            print(f"F1 Micro: {metrics['F1_micro']:.4f}")
            print(f"F1 Macro: {metrics['F1_macro']:.4f}")
            print(f"AUC Micro: {metrics['AUC_micro']:.4f}")
            print(f"AUC Macro: {metrics['AUC_macro']:.4f}")
            print(f"Subset Accuracy: {metrics['Subset_Accuracy']:.4f}")
            eval_end_time = time.time()
            eval_duration = eval_end_time - eval_start_time
            print(f'Evaluation time：{eval_duration}')
# ------------------------------------------------------------------------------------------------------------------------------#

# -------------------------------------------PRINT----------------------------------------------------------------------------#
        scheduler.step()
        print(f'<<---------------------------------train epoch {epoch} end --------------------------------->>')
    print('------------------------Lora Fine-Tune End--------------------------')
# -----------------------------------------------------------------------------------------------------------------------#




def calculate_metrics(all_pred_labels, all_opt_labels, device='cuda:0'):

    all_pred_labels = all_pred_labels.to(device)
    all_opt_labels = all_opt_labels.to(device)


    y_pred = all_pred_labels.cpu().numpy().flatten()
    y_true = (all_opt_labels > 0.1).float().cpu().numpy().flatten()


    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    try:
        auc_micro = roc_auc_score(y_true, y_pred, average='micro')
        auc_macro = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:

        auc_micro = 0.0
        auc_macro = 0.0
        print("AUC cannot be calculated; there may be a scenario where only one class exists.")

    subset_accuracy = torch.all(all_pred_labels == (all_opt_labels > 0.1).float(), dim=1).float().mean()
    print(classification_report(y_true, y_pred, zero_division=0))
    return {
        'F1_micro': f1_micro,
        'F1_macro': f1_macro,
        'AUC_micro': auc_micro,
        'AUC_macro': auc_macro,
        'Subset_Accuracy': subset_accuracy.item()
    }
