import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy
import os
from flask import Flask, request, jsonify
import io
import base64
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from torch_geometric.data import Data, Batch
from loss import EnhancedVACCLoss
from sql_dataset import TrainDataset
from CombinedDataset import CombinedDataset
from model import Llama4SQA
from Embedding import PlanGAT, TimeLSTM
from Combine_model import CombinedModel
from sklearn.metrics import f1_score, roc_auc_score
import nltk
from nltk.tokenize import word_tokenize
import hashlib
from sklearn.metrics import f1_score, roc_auc_score, classification_report
app = Flask(__name__)


server_model = None
server_tokenizer = None
text_model = None
args = None
round_num = 0
client_deltas = []
num_clients = 4

def load_text_model(vocab_size, embedding_dim, load_path, device):

    text_model = nn.Embedding(vocab_size, embedding_dim).to(device)
    text_model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))
    text_model.eval()
    print(f"Text embedding model loaded: {load_path}")
    return text_model


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
            return torch.zeros(vector_size, device=args.device)

    if not isinstance(text, str):
        text = str(text)

    tokens = word_tokenize(text.lower())
    indices = [hash_word(word, model.num_embeddings) for word in tokens]

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
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    precision_1 = report_dict.get('1.0', {}).get('precision', 0.0)
    print(f"R-ACC: {precision_1:.4f}")
    return {
        'F1_micro': f1_micro,
        'F1_macro': f1_macro,
        'AUC_micro': auc_micro,
        'AUC_macro': auc_macro,
        'Subset_Accuracy': subset_accuracy.item(),
        'R_ACC': precision_1
    }


class AdaptedCombinedModel(nn.Module):

    def __init__(self, pretrained_combined_model, num_labels=9, common_dim=4096):
        super().__init__()

        self.graph_model = pretrained_combined_model.graph_model
        self.time_model = pretrained_combined_model.time_model
        self.LLM_base = pretrained_combined_model.LLM_model.llama
        self.fusion_layer = pretrained_combined_model.fusion_layer

        for param in self.graph_model.parameters():
            param.requires_grad = False
        for param in self.time_model.parameters():
            param.requires_grad = False
        for param in self.LLM_base.parameters():
            param.requires_grad = False


        self.fusion_type = pretrained_combined_model.fusion_type
        self.proj_text = pretrained_combined_model.text_proj
        self.proj_graph = pretrained_combined_model.graph_proj
        self.proj_time = pretrained_combined_model.time_proj
        for param in self.proj_text.parameters():
            param.requires_grad = False
        for param in self.proj_graph.parameters():
            param.requires_grad = False
        for param in self.proj_time.parameters():
            param.requires_grad = False
        self._init_weights(pretrained_combined_model)

    def _init_weights(self, pretrained_combined_model):
        self.classifiers = nn.ModuleList()
        for pretrained_cls in pretrained_combined_model.LLM_model.classifiers:
            self.classifiers.append(pretrained_cls)

    def forward(self, text_inputs, graph_data, time_series):
        with torch.no_grad():

            text_features = self.proj_text(text_inputs)


            graph_features = self.graph_model(graph_data)
            graph_features = self.proj_graph(graph_features)


            time_features = self.time_model(time_series)
            time_features = time_features.squeeze(0)
            time_features = self.proj_time(time_features)


            if self.fusion_type == 'concat':
                fused_features = torch.cat([text_features, graph_features, time_features], dim=1)
            else:
                fused_features = text_features + graph_features + time_features
            fusion_embeding = self.fusion_layer(fused_features)

            llm_input_embeds = fusion_embeding.unsqueeze(0).unsqueeze(1)
            outputs = self.LLM_base(inputs_embeds=llm_input_embeds,
                                    output_hidden_states=True,
                                    return_dict=True
                                    )
            llm_features = outputs.hidden_states[-1].mean(dim=1)


        logits = [classifier(llm_features) for classifier in self.classifiers]
        logits = torch.cat(logits, dim=1)
        return logits


def load_combined_model(load_path, args, device):

    print(f"Loading federated model weights from {load_path}...")


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

    llama = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    llama_lora = get_peft_model(llama, lora_config)
    llm_model = Llama4SQA(llama_lora, num_labels=9).to(device)


    graph_model = PlanGAT(in_channels=11, hidden_channels=128, out_channels=256).to(device)


    time_model = TimeLSTM(input_size=1, hidden_size=64, q=7, output_dim=40, num_layers=1).to(device)


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


    print("Loading the state dictionary into CPU memory...")
    model_state = torch.load(load_path, map_location='cpu', weights_only=True)

    print("Loading the state dictionary into the model...")
    combined_model.load_state_dict(model_state['combined_model_state'])

    print(f"The joint model weights have been successfully loaded from the {model_state['epoch']}th training cycle.")


    del model_state
    torch.cuda.empty_cache()

    return combined_model, tokenizer


def scale_and_aggregate_weights(client_deltas):

    avg_deltas = {}
    for key in client_deltas[0]:
        avg_deltas[key] = torch.stack([delta[key].to(torch.float32) for delta in client_deltas]).mean(dim=0)


    scaled_deltas = {}
    for key in avg_deltas:
        scale_factor = 1.0
        scaled_deltas[key] = avg_deltas[key] * scale_factor
    return scaled_deltas


def evaluate_server_model():

    global server_model, text_model, args

    print("\n--- Evaluating Server Model ---")

    test_data = CombinedDataset(args, 'test')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    server_model.eval()
    text_model.eval()

    all_pred_opt = []
    all_opt_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Server Evaluation"):
            prompt, labels, opt_labels, graph_data, time_series = batch


            text_embeddings = text_to_vector(prompt, text_model, args)
            text_embeddings = text_embeddings.to(args.device)

            labels = labels.to(args.device)
            opt_labels = opt_labels.to(args.device)


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


            logits = server_model(
                text_inputs=text_embeddings,
                graph_data=graph_batch,
                time_series=time_series
            )

            probs = torch.sigmoid(logits)
            all_pred_opt.append(probs)
            all_opt_labels.append(opt_labels)


    all_pred_opt = torch.cat(all_pred_opt, dim=0)
    all_opt_labels = torch.cat(all_opt_labels, dim=0)

    mse = torch.mean((all_pred_opt - all_opt_labels) ** 2).item()
    mae = torch.mean(torch.abs(all_pred_opt - all_opt_labels)).item()

    all_multilabel_pred = (all_pred_opt > 0.5).float()
    all_multilabel_true = (all_opt_labels > 0.1).float()
    right_label_all = torch.where(all_multilabel_true == all_multilabel_pred, 1, 0).sum()
    vacc = right_label_all / (len(test_loader.dataset) * 9)

    metrics = calculate_metrics(all_multilabel_pred, all_opt_labels, args.device)

    print(f"\nServer Model Performance:")
    print(f"V-ACC: {metrics['F1_micro']:.4f}")
    print(f"F1: {metrics['F1_macro']:.4f}")
    print(f"AUC: {metrics['AUC_micro']:.4f}")
    print(f"R-ACC:{metrics['R_ACC']:.4f}")

    return metrics


def serialize_tensor(tensor):

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def deserialize_tensor(b64_string, device):

    buffer = io.BytesIO(base64.b64decode(b64_string))
    tensor = torch.load(buffer, map_location=device)
    return tensor


def serialize_state_dict(state_dict):
    serialized = {}
    for key, tensor in state_dict.items():
        serialized[key] = serialize_tensor(tensor)
    return serialized


def deserialize_state_dict(serialized, device):
    state_dict = {}
    for key, b64_string in serialized.items():
        state_dict[key] = deserialize_tensor(b64_string, device)
    return state_dict


def init_server(config):
    global server_model, server_tokenizer, text_model, args

    args = config


    text_model = load_text_model(
        vocab_size=10000,
        embedding_dim=64,
        load_path=args.text,
        device=args.device
    )


    server_path = args.sever_model
    pretrained_server_model, server_tokenizer = load_combined_model(
        load_path=server_path, args=args, device=args.device
    )


    server_model = AdaptedCombinedModel(
        pretrained_server_model, num_labels=9, common_dim=4096
    ).to(args.device)

    print("✓ Server model initialization completed")
    print("Evaluating the Basic Server Model")
    metrics = evaluate_server_model()
    return True



@app.route('/init', methods=['POST'])
def initialize_server():
    config = request.json


    class Args:
        pass

    args_obj = Args()
    for key, value in config.items():
        setattr(args_obj, key, value)


    if hasattr(args_obj, 'cuda') and args_obj.cuda and torch.cuda.is_available():
        args_obj.device = f"cuda:{args_obj.cuda_id}" if hasattr(args_obj, 'cuda_id') else "cuda:0"
    else:
        args_obj.device = "cpu"

    success = init_server(args_obj)

    if success:
        return jsonify({
            'status': 'success',
            'message': 'Server initialization successful'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Server initialization failed'
        }), 500


@app.route('/get_classifier_weights', methods=['GET'])
def get_classifier_weights():
    global server_model

    if server_model is None:
        return jsonify({
            'status': 'error',
            'message': 'The server model has not been initialized.'
        }), 400


    classifier_weights = server_model.classifiers.state_dict()
    serialized_weights = serialize_state_dict(classifier_weights)

    return jsonify({
        'status': 'success',
        'classifier_weights': serialized_weights
    })



import time
from datetime import datetime



@app.route('/submit_delta', methods=['POST'])
def submit_delta():

    global client_deltas, round_num, num_clients

    receive_time = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Received weight increment submission request")

    data = request.json
    client_id = data['client_id']
    serialized_delta = data['delta']
    client_send_time = data.get('send_timestamp', receive_time)


    network_time = receive_time - client_send_time


    processing_start = time.time()
    delta = deserialize_state_dict(serialized_delta, args.device)
    processing_time = time.time() - processing_start


    total_size = sum(len(v.encode('utf-8')) for v in serialized_delta.values())
    size_mb = total_size / (1024 * 1024)

    print(f"Received weight increment from client {client_id}")
    print(f"  - Network transmission time: {network_time:.3f}s")
    print(f"  - Server processing time: {processing_time:.3f}s")
    print(f"  - Data size: {size_mb:.2f} MB")
    print(f"  - Transfer speed:  {size_mb / network_time:.2f}MB/s" if network_time > 0.001 else "Transfer speed:  >1000MB/s")

    print(f"Number of clients required by the server{num_clients}")


    if len(client_deltas) < client_id + 1:
        client_deltas.extend([None] * (client_id + 1 - len(client_deltas)))
    client_deltas[client_id] = delta
    start_eval = time.time()

    if len(client_deltas) >= num_clients and all(delta is not None for delta in client_deltas[:num_clients]):
        print("All client increments have been received. Starting aggregation...")

        scaled_deltas = scale_and_aggregate_weights(client_deltas[:num_clients])


        server_weights = server_model.classifiers.state_dict()
        for key in server_weights:
            server_weights[key] = server_weights[key] + scaled_deltas[key]
        server_model.classifiers.load_state_dict(server_weights)

        print("Evaluate the updated server model")

        metrics = evaluate_server_model()
        end_eval = time.time()
        eval_time = end_eval - start_eval
        client_deltas = []
        round_num += 1

        return jsonify({
            'status': 'success',
            'message': 'The increment has been submitted and completed aggregation.',
            'round_completed': True,
            'metrics': metrics,
            'round_num': round_num,
            'communication_time': network_time,
            'data_size_mb': size_mb,
            'eval_time':eval_time
        })
    else:
        return jsonify({
            'status': 'success',
            'message': 'The increment has been submitted and is awaiting other clients.',
            'round_completed': False,
            'communication_time': network_time,
            'data_size_mb': size_mb
        })


@app.route('/set_num_clients', methods=['POST'])
def set_num_clients():

    global num_clients

    data = request.json
    num_clients = data['num_clients']

    return jsonify({
        'status': 'success',
        'message': f'The expected number of clients has been set to {num_clients}'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)