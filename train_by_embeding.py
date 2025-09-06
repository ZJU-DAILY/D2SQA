import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, classification_report
import torch.optim as optim
from torch.optim import lr_scheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from torch_geometric.data import Data, Batch
import hashlib
import torch.nn.init as init

# 自定义模块导入
from loss import EnhancedVACCLoss
from Embedding import PlanGAT, TimeLSTM
from CombinedDataset import CombinedDataset
from Combine_model import CombinedModel
from model import Llama4SQA
from nltk.tokenize import word_tokenize
import nltk

def train_combined_model(args):
    model_path = args.pre_path + '/models/LLAMA3.1-8B-Instruct/'
    print(f"Using the 8B model, dataset:{args.dataset}")
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
    model = Llama4SQA(llama_lora, num_labels=9).to(args.device)
    print('------------------------Joint model training begins--------------------------')

    vocab_size = 10000
    embedding_dim = 64


    text_model = nn.Embedding(vocab_size, embedding_dim)
    text_dropout = nn.Dropout(p=0.3)
    init.xavier_uniform_(text_model.weight, gain=1.0)

    graph_model = PlanGAT(in_channels=11, hidden_channels=128, out_channels=256).to(args.device)


    time_model = TimeLSTM(input_size=1, hidden_size=64, q=7, output_dim=40, num_layers=1)

    combined_model = CombinedModel(
        LLM_model=model,
        graph_model=graph_model,
        time_model=time_model,
        fusion_type='add',
        text_dim=64,
        graph_dim=256,
        time_dim=280,
        fusion_dim=4096
    ).to(args.device)


    train_data = CombinedDataset(args, 'train')
    val_data = CombinedDataset(args,  'val')
    test_data = CombinedDataset(args, 'test')
    dataloader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    dataloader_val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    dataloader_test = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    params = list(combined_model.parameters()) + list(text_model.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    alpha = 1
    beta = 0
    gamma = 0

    loss_fn = EnhancedVACCLoss(
        epsilon=0.1,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    ).to(args.device)
# -----------------------------------------------------------------------------------------------------------------------#

    test_len = len(dataloader_test)
    combined_model.eval()
    text_model.eval()
    text_dropout.eval()
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

            text_embeddings = text_to_vector(prompt, text_model,text_dropout,args)
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
            combined_model.to("cuda:0")
            logits = combined_model(
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
# -----------------------------------------------------------------------------------------------------------------------#
    best_vacc = 0
    best_auc = 0
    best_f1 = 0
    # 6. 训练循环
    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'<<------------------- epoch {epoch}/{args.epochs} , lr: {current_lr}------------------>>')

        combined_model.train()
        text_model.train()
        text_dropout.train()
        total_loss = 0

        for batch_idx, batch in enumerate(tqdm(dataloader_train, leave=True)):
            prompt, labels, opt_labels, graph_data, time_series = batch

            text_embeddings = text_to_vector(prompt, text_model,text_dropout,args)
            text_embeddings = text_embeddings.to(args.device)

            labels, opt_labels = labels.to(args.device), opt_labels.to(args.device)


            if isinstance(graph_data, dict):

                if "batch" not in graph_data:
                    x = graph_data["x"].to(args.device)
                    edge_index = graph_data["edge_index"].to(args.device)


                    if x.dim() == 3 and x.shape[0] == 1:
                        x = x.squeeze(0)
                    if edge_index .dim() == 3 and edge_index .shape[0] == 1:
                        edge_index  = edge_index .squeeze(0)

                    data = Data(x=x, edge_index=edge_index)


                    graph_batch = Batch.from_data_list([data])
                else:

                    graph_batch = graph_data.to(args.device)
            else:
                graph_batch = graph_data.to(args.device)


            time_series = time_series.to(args.device)

            text_embeddings = text_embeddings.to("cuda:0")
            combined_model.to("cuda:0")
            outputs = combined_model(
                text_inputs=text_embeddings,
                graph_data=graph_batch,
                time_series=time_series
            )


            loss,binary_loss, regression_loss, threshold_loss = loss_fn(
                outputs,labels, opt_labels
            )


            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(combined_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()


        avg_epoch_loss = total_loss / len(dataloader_train)
        print(f'Epoch {epoch} Completed, Average loss: {avg_epoch_loss:.4f}')


        test_len = len(dataloader_test)
        combined_model.eval()
        text_model.eval()
        text_dropout.eval()
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

                text_embeddings = text_to_vector(prompt, text_model,text_dropout,args)
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
                combined_model.to("cuda:0")
                logits = combined_model(
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
            print(f'Epoch {epoch} - Test MSE: {mse}, MAE: {mae}, V-ACC: {vacc},top1acc:{top1acc},mcacc:{mcacc},top1IR:{top1IR}')

            metrics = calculate_metrics(all_multilabel_pred, all_opt_labels)
            print("\nNewly added evaluation metrics:")
            print(f"F1 Micro: {metrics['F1_micro']:.4f}")
            print(f"F1 Macro: {metrics['F1_macro']:.4f}")
            print(f"AUC Micro: {metrics['AUC_micro']:.4f}")
            print(f"AUC Macro: {metrics['AUC_macro']:.4f}")
            print(f"Subset Accuracy: {metrics['Subset_Accuracy']:.4f}")
            # 对比并保存满足条件的模型
            # if vacc > 0.6656 and metrics['AUC_micro'] > 0.5313 and metrics['F1_macro'] > 0.5281:
            #     save_path = args.pre_path + f"/combined_model/droup0.2_tpc_ds_llama_8B_{vacc}_{metrics['AUC_micro']}.pth"

            #     save_combined_model(combined_model, text_model, save_path, epoch)
            #     print(f"The model has been saved to: {save_path}")
            if vacc > best_vacc and metrics['AUC_micro'] > best_auc and metrics['F1_macro'] > best_f1 and (epoch % 10 == 0):
                best_vacc = vacc
                best_auc = metrics['AUC_micro']
                best_f1 = metrics['F1_macro']
                save_path = args.pre_path + f"/combined_model/{args.dataset}_llama_8B_{vacc}_{metrics['AUC_micro']}.pth"
                save_combined_model(combined_model, text_model, save_path, epoch)
                print(f"The model has been saved to: {save_path}")

        scheduler.step()
        print(f'<<-------------------Training epoch {epoch} completed------------------>>')

    print('------------------------Joint model training completed--------------------------')


def save_combined_model(combined_model, text_model, save_path, epoch):

    model_state = {
        'epoch': epoch,
        'combined_model_state': combined_model.state_dict()
    }
    torch.save(model_state, save_path)
    print(f"The joint model has been saved to: {save_path} (Epoch {epoch})")

    text_save_path = save_path.replace("llama_8B", "text_llama_8B")
    save_text_model(text_model, text_save_path)




def hash_word(word, vocab_size):

    hash_obj = hashlib.md5(word.encode('utf-8'))
    hash_int = int(hash_obj.hexdigest()[:8], 16)
    return (hash_int % (vocab_size - 1)) + 1
def text_to_vector(text, model,dropout_layer, args, vector_size=64):

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


    indices = [hash_word(word, model.num_embeddings) for word in tokens]

    if not indices:
        return torch.zeros(vector_size, device=args.device)


    indices_tensor = torch.tensor(indices, dtype=torch.long, device=model.weight.device)


    vectors = model(indices_tensor)
    vectors = dropout_layer(vectors)

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
def save_text_model(text_model, save_path):
    torch.save(text_model.state_dict(), save_path)
    print(f"The text embedding model has been saved to: {save_path}")