import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, jaccard_score
import torch.optim as optim
from torch.optim import lr_scheduler
import gensim
from gensim.models import Word2Vec
import os
import pandas as pd
from loss import EnhancedVACCLoss
from Embedding import PlanGAT, TimeLSTM
from CombinedDataset import CombinedDataset
from Combine_model import CombinedModel
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
import numpy as np
from torch.optim import lr_scheduler
from loss import EnhancedVACCLoss
from subdateset import CustomSubset
import gensim
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from torch_geometric.data import Data, Batch
from distillation import dynamic_loss, compute_jsd_sigmoid
import torch.nn.functional as F
from model_prompt import Llama4SQA_prompt
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import classification_report
import hashlib
def train_distillation(args):

    device = args.device
    load_path =  args.teacher
    teacher_model, tokenizer_teacher = load_combined_model(load_path=load_path,args=args,device=device)
    student_model, tokenizer = load_stu_model(args, device)

    teacher_model.to(device)
    student_model.to(device)
    teacher_model.eval()


    train_data = CombinedDataset(args, 'train')
    val_data = CombinedDataset(args, 'val')
    test_data = CombinedDataset(args, 'test')
    train_data_student = TrainDataset(args, tokenizer, 'train')
    val_data_student = TrainDataset(args, tokenizer, 'val')
    test_data_student = TrainDataset(args, tokenizer, 'test')

    dataloader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    dataloader_val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    dataloader_test = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    dataloader_train_stu = DataLoader(train_data_student, batch_size=args.batch_size, shuffle=False)
    dataloader_val_stu = DataLoader(val_data_student, batch_size=args.batch_size, shuffle=False)
    dataloader_test_stu = DataLoader(test_data_student, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


    epsilon = 0.1
    alpha = 1
    beta = 0
    gamma = 0

    loss_fn = EnhancedVACCLoss(
        epsilon=0.1,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    ).to(args.device)

    tau = 0.3
    vocab_size = 10000
    embedding_dim = 64


    text_model = load_text_model(
        vocab_size=10000,
        embedding_dim=64,
        load_path=args.text,
        device=args.device
    )


    test_len = len(dataloader_test)
    teacher_model.eval()
    text_model.eval()
    with torch.no_grad():
        top_1_cor = 0
        rank_pred = []
        rank_true = []
        top1_valid_sum = 0
        all_right_cnt = 0
        all_pred_opt = []
        all_opt_labels = []
        for batch_idx, batch in enumerate(tqdm(dataloader_test, leave=True)):
            prompt, labels, opt_labels, graph_data, time_series = batch

            text_embeddings = text_to_vector(prompt, text_model, args)
            text_embeddings = text_embeddings.to(args.device)

            labels, opt_labels = labels.to(args.device), opt_labels.to(args.device)

            if isinstance(graph_data, dict):

                if "batch" not in graph_data:
                    x = graph_data["x"].to(args.device)
                    edge_index = graph_data["edge_index"].to(args.device)


                    if x.dim() == 3 and x.shape[0] == 1:
                        x = x.squeeze(0)
                    if edge_index.dim() == 3 and edge_index.shape[0] == 1:
                        edge_index = edge_index.squeeze(0)

                    data = Data(x=x, edge_index=edge_index)


                    graph_batch = Batch.from_data_list([data])
                else:

                    graph_batch = graph_data.to(args.device)
            else:
                graph_batch = graph_data.to(args.device)


            time_series = time_series.to(args.device)

            text_embeddings = text_embeddings.to("cuda:0")
            teacher_model.to("cuda:0")
            logits = teacher_model(
                text_inputs=text_embeddings,
                graph_data=graph_batch,
                time_series=time_series
            )
            probs = torch.sigmoid(logits)
            # probs = logits
            pred_opt = probs
            sorted_time_index = torch.argsort(pred_opt, dim=1, descending=True)
            label_sorted_time_index = torch.argsort(opt_labels, dim=1, descending=True)

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

        all_pred_opt = torch.cat(all_pred_opt, dim=0)
        all_opt_labels = torch.cat(all_opt_labels, dim=0)

        mse = torch.mean((all_pred_opt - all_opt_labels) ** 2).item()
        mae = torch.mean(torch.abs(all_pred_opt - all_opt_labels)).item()

        all_multilabel_pred = (all_pred_opt > 0.5).float()
        all_multilabel_true = (all_opt_labels > 0.1).float()
        right_label_all = torch.where(all_multilabel_true == all_multilabel_pred, 1, 0).sum()
        vacc = right_label_all / test_len / 9
        top1acc = top_1_cor / float(test_len)
        mcacc = all_right_cnt / float(test_len)
        top1IR = top1_valid_sum / float(test_len)
        print(f'Baseline---Test MSE: {mse}, MAE: {mae}, V-ACC: {vacc},top1acc:{top1acc},mcacc:{mcacc},top1IR:{top1IR}')
        metrics = calculate_metrics(all_multilabel_pred, all_opt_labels)
        print("\nNewly added evaluation metrics:")
        print(f"F1 Micro: {metrics['F1_micro']:.4f}")
        print(f"F1 Macro: {metrics['F1_macro']:.4f}")
        print(f"AUC Micro: {metrics['AUC_micro']:.4f}")
        print(f"AUC Macro: {metrics['AUC_macro']:.4f}")
        print(f"Subset Accuracy: {metrics['Subset_Accuracy']:.4f}")
    #--------------------------------------------------------------------------------------------------------------------------------#
    for epoch in range(args.epochs):
        student_model.train()
        total_loss = 0
        current_lr = optimizer.param_groups[0]['lr']
        print(f'<<-------- Distill epoch {epoch}/{args.epochs},lr:{current_lr}-------->>')


        for batch_idx, (batch, batch_student) in enumerate(tqdm(zip(dataloader_train, dataloader_train_stu), leave=True)):
            prompt, labels, opt_labels, graph_data, time_series = batch
            input_ids, att_mask, labels_stu, opt_labels_stu = batch_student
            input_ids, att_mask, labels_stu, opt_labels_stu = input_ids.to(device), att_mask.to(device), labels_stu.to(device), opt_labels_stu.to(device)
            opt_labels = opt_labels.to(device)

            if not torch.equal(opt_labels, opt_labels_stu):
                mismatch_pos = (opt_labels != opt_labels_stu).nonzero(as_tuple=False)
                print(f"[Batch {batch_idx}] ❌ Inconsistencies, top 5 difference indexes: {mismatch_pos[:5].tolist()}")

            text_embeddings = text_to_vector(prompt, text_model, args)
            text_embeddings = text_embeddings.to(args.device)

            labels, opt_labels = labels.to(args.device), opt_labels.to(args.device)


            if isinstance(graph_data, dict):

                if "batch" not in graph_data:
                    x = graph_data["x"].to(args.device)
                    edge_index = graph_data["edge_index"].to(args.device)


                    if x.dim() == 3 and x.shape[0] == 1:
                        x = x.squeeze(0)
                    if edge_index.dim() == 3 and edge_index.shape[0] == 1:
                        edge_index = edge_index.squeeze(0)

                    data = Data(x=x, edge_index=edge_index)


                    graph_batch = Batch.from_data_list([data])
                else:

                    graph_batch = graph_data.to(args.device)
            else:
                graph_batch = graph_data.to(args.device)


            time_series = time_series.to(args.device)

            text_embeddings = text_embeddings.to("cuda:0")

            # -----------------------------------------------

            with torch.no_grad():
                teacher_logits = teacher_model(
                    text_inputs=text_embeddings,
                    graph_data=graph_batch,
                    time_series=time_series
                )

                teacher_probs = torch.sigmoid(teacher_logits)
            student_logits = student_model(input_ids=input_ids, attention_mask=att_mask)
            student_probs = torch.sigmoid(student_logits)
            jsd = compute_jsd_sigmoid(student_probs, teacher_probs)
            alpha = torch.clamp(torch.exp(-jsd.detach() / 0.15), min=0.1)
            kl_loss = F.binary_cross_entropy(student_probs, teacher_probs)
            task_loss, _, _, _ = loss_fn(student_logits, labels, opt_labels)
            loss = dynamic_loss(alpha, kl_loss, task_loss)
            # -----------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_epoch_loss = total_loss / len(dataloader_train)
        print(f'<<-------- Epoch {epoch} done, Avg Loss: {avg_epoch_loss:.4f} -------->>')

        test_len = len(dataloader_test_stu)
        student_model.eval()
        with torch.no_grad():
            all_pred_opt = []
            all_opt_labels = []
            for batch_idx, batch in enumerate(tqdm(dataloader_test_stu, leave=True)):
                input_ids, att_mask, labels, opt_labels = batch
                input_ids, att_mask, labels, opt_labels = input_ids.to(device), att_mask.to(device), labels.to(
                    device), opt_labels.to(device)
                logits = student_model(input_ids=input_ids, attention_mask=att_mask)
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
            all_multilabel_true = (all_opt_labels > 0.1).float()
            right_label_all = torch.where(all_multilabel_true == all_multilabel_pred, 1, 0).sum()
            vacc = right_label_all / test_len / 9
            metrics = calculate_metrics(all_multilabel_pred, all_opt_labels)
            print("\nNewly added evaluation metrics:")
            print(f"F1 Micro: {metrics['F1_micro']:.4f}")
            print(f"F1 Macro: {metrics['F1_macro']:.4f}")
            print(f"AUC Micro: {metrics['AUC_micro']:.4f}")
            print(f"AUC Macro: {metrics['AUC_macro']:.4f}")
            print(f"Subset Accuracy: {metrics['Subset_Accuracy']:.4f}")
            if vacc > 0.6656 and metrics['AUC_micro'] > 0.5313 and metrics['F1_macro'] > 0.5281:
                save_dir = args.pre_path + f"/combined_model/llama8B_{args.dataset}_student_lora_weights_{epoch}_{vacc}_{metrics['AUC_micro']}"
                student_model.llama.save_pretrained(os.path.join(save_dir, "lora_adapter"))
                classifiers_state = student_model.classifiers.state_dict()
                torch.save(
                    classifiers_state,
                    os.path.join(save_dir, "classifiers_weights.pt")
                )
                print(f"The best model has been saved to {save_dir}")
        scheduler.step()
    print('---------------- Distillation Finished ----------------')




def load_combined_model(load_path, args, device):
    print(f"Loading federated model weights from {load_path}...")

    # 1. 重建LLM模型
    model_path = args.pre_path + '/models/LLAMA3.1-8B-Instruct/'
    tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    tokenizer.pad_token = tokenizer.eos_token
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    llama = AutoModelForCausalLM.from_pretrained(model_path)
    llama_lora = get_peft_model(llama, lora_config)
    llm_model = Llama4SQA(llama_lora, num_labels=9).to(device)


    graph_model = PlanGAT(in_channels=11, hidden_channels=128, out_channels=256).to(device)


    time_model = TimeLSTM(input_size=1, hidden_size=64, q=7, output_dim=40, num_layers=1)

    combined_model = CombinedModel(
        LLM_model=llm_model,
        graph_model=graph_model,
        time_model=time_model,
        fusion_type='add',
        text_dim=64,
        graph_dim=256,
        time_dim=280,
        fusion_dim=4096
    ).to(device)


    try:
        model_state = torch.load(load_path,weights_only=True)
    except:
        model_state = torch.load(load_path, map_location=torch.device('cpu'),weights_only=True)

    combined_model.load_state_dict(model_state['combined_model_state'])
    print(f"The joint model weights have been successfully loaded from the {model_state['epoch']}th training cycle.")

    return combined_model, tokenizer
def load_stu_model(args, device):

    model_path = args.pre_path + '/models/llama-1B/'
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
    llama = AutoModelForCausalLM.from_pretrained(model_path)
    llama_lora = get_peft_model(llama, lora_config)
    llm_model = Llama4SQA_prompt(llama_lora, num_labels=9).to(device)

    return llm_model, tokenizer
def hash_word(word, vocab_size):

    hash_obj = hashlib.md5(word.encode('utf-8'))
    hash_int = int(hash_obj.hexdigest()[:8], 16)
    return (hash_int % (vocab_size - 1)) + 1
def text_to_vector(text, model, args, vector_size=64):

    nltk.data.path.append(args.pre_path + '/models/nltk_data')


    if isinstance(text, tuple):
        if len(text) > 0:
            text = text[0]
        else:
            print("Use a prompt with all zeros")
            return torch.zeros(vector_size, device=args.device)


    if not isinstance(text, str):
        text = str(text)


    tokens = word_tokenize(text.lower())


    indices = [hash_word(word,model.num_embeddings) for word in tokens]

    if not indices:
        return torch.zeros(vector_size, device=args.device)


    indices_tensor = torch.tensor(indices, dtype=torch.long, device=model.weight.device)


    vectors = model(indices_tensor)


    return torch.mean(vectors, dim=0)
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
def load_text_model(vocab_size, embedding_dim, load_path, device):

    text_model = nn.Embedding(vocab_size, embedding_dim).to(device)
    text_model.load_state_dict(torch.load(load_path, map_location=device,weights_only=True))
    text_model.eval()
    print(f"Text embedding model loaded: {load_path}")
    return text_model

