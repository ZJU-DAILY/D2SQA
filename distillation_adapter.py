import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import LlamaForCausalLM, AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, get_peft_model_state_dict
from sql_dataset import TrainDataset
from prompt import construct_prompt_lora
from tqdm import tqdm
import os
import pandas as pd
from loss import EnhancedVACCLoss
from Embedding import PlanGAT, TimeLSTM
from CombinedDataset import CombinedDataset
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


class AttentionPool(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPool, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, inputs):
        scores = self.projection(inputs)  # [batch, seq_len, 1]
        attention_weights = torch.softmax(scores, dim=1)
        weighted_sum = torch.sum(inputs * attention_weights, dim=1)
        return weighted_sum


class Llama4SQA_sever(nn.Module):
    def __init__(self, llama_lora_model, num_labels=9):
        super().__init__()
        self.llama = llama_lora_model
        self.classifiers = nn.ModuleList([
            nn.Linear(self.llama.config.hidden_size, 1) for _ in range(num_labels)
        ])
        self.attentionPool = AttentionPool(self.llama.config.hidden_size)
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        for classifier in self.classifiers:
            nn.init.xavier_normal_(classifier.weight)
            if classifier.bias is not None:
                nn.init.zeros_(classifier.bias)

    def forward(self, inputs_embeds):
        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True
        )

        last_hidden = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]

        # Since the teacher model's input is a single fused vector, seq_len=1, so mean(dim=1) is equivalent to squeeze(1)
        # pooled has shape [batch_size, hidden_size]
        pooled = last_hidden.mean(dim=1)  # <--- This is h_server

        # Classification
        logits = torch.cat([
            classifier(pooled) for classifier in self.classifiers
        ], dim=1)

        # MODIFIED: Return logits and pooled features
        return logits, pooled


class CombinedModel_sever(nn.Module):
    def __init__(self, LLM_model, graph_model, time_model, fusion_type='concat',
                 text_dim=512, graph_dim=64, time_dim=64, fusion_dim=2560):
        super().__init__()
        # Predefined linear layers to avoid dynamic creation
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.graph_proj = nn.Linear(graph_dim, fusion_dim)
        self.time_proj = nn.Linear(time_dim, fusion_dim)
        # Save sub-models
        self.LLM_model = LLM_model
        self.graph_model = graph_model
        self.time_model = time_model
        self.fusion_dim = fusion_dim
        # Fusion type configuration
        self.fusion_type = fusion_type

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.1),  # Add Dropout layer, default dropout rate is 0.5, can be adjusted
            nn.Linear(fusion_dim // 2, fusion_dim)
        )

    def forward(self, text_inputs, graph_data, time_series):
        # 1. Process text input
        text_embedding = text_inputs
        text_embedding = text_embedding.to("cuda:0")
        # 2. Process graph data
        graph_embedding = self.graph_model(graph_data)
        graph_embedding = graph_embedding.to("cuda:0")
        # 3. Process time series
        time_embedding = self.time_model(time_series)
        time_embedding = time_embedding.squeeze(0)  # Remove batch dimension
        time_embedding = time_embedding.to("cuda:0")
        # 4. Fuse different modality embeddings
        if self.fusion_type == 'concat':
            fusion_input = torch.cat([text_embedding, graph_embedding, time_embedding], dim=0)
        elif self.fusion_type == 'add':
            # Adjust each embedding to fusion_dim
            text_emb = self.text_proj(text_embedding)
            graph_emb = self.graph_proj(graph_embedding)
            time_emb = self.time_proj(time_embedding)
            fusion_input = text_emb + graph_emb + time_emb
        else:
            raise ValueError("Unsupported fusion type")

        # 5. Final output
        fusion_embeding = self.fusion_layer(fusion_input)

        # 1. Ensure fusion embedding is 3D: [batch_size, seq_length, hidden_size]
        fusion_embeding = fusion_embeding.unsqueeze(0).unsqueeze(1)

        output, h_server = self.LLM_model(inputs_embeds=fusion_embeding)
        return output, h_server

    def _resize_embedding(self, embedding, target_dim):
        """Resize embedding dimension to match target dimension"""
        if embedding.shape[0] == target_dim:
            return embedding
        return nn.Linear(embedding.shape[0], target_dim)(embedding)


# Modified Llama4SQA_prompt class (student model)

class Llama4SQA_prompt_client(nn.Module):
    # Modified __init__
    # projection_dim is now the hidden_size of the teacher LLM
    def __init__(self, llama_lora_model, num_labels=9, projection_dim=4096):
        super().__init__()

        self.hidden_size = llama_lora_model.config.hidden_size  # Student model's hidden_size (e.g., 2048)
        self.llama = llama_lora_model

        # NEW: Adapter layer to map student's pooled (h_client) to teacher's pooled (h_server) dimension
        # Input: student's hidden_size, Output: teacher's hidden_size (projection_dim)
        self.projection_layer = nn.Linear(self.hidden_size, projection_dim)

        # Classifier still acts on student's original pooled features, or acts on the projected features
        # Classifiers act on the projected features. This allows the projection layer to receive gradients from the task loss.
        self.classifiers = nn.ModuleList([
            nn.Linear(projection_dim, 1) for _ in range(num_labels)
        ])

        self._init_weights()

    def _init_weights(self):
        # Initialize projection layer and classifiers
        nn.init.xavier_normal_(self.projection_layer.weight)
        if self.projection_layer.bias is not None: nn.init.zeros_(self.projection_layer.bias)
        for classifier in self.classifiers:
            nn.init.xavier_normal_(classifier.weight)
            if classifier.bias is not None: nn.init.zeros_(classifier.bias)

    # Modified forward
    def forward(self, input_ids, attention_mask):
        outputs = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        last_hidden = outputs.hidden_states[-1]

        # Mean Pool
        attention_mask_expanded = attention_mask.unsqueeze(-1)
        masked_hidden = last_hidden * attention_mask_expanded
        sum_hidden = masked_hidden.sum(dim=1)
        valid_token_counts = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_hidden / valid_token_counts  # <--- h_client [batch_size, student_hidden_size]

        # NEW: Project h_client to get h_client_adapted
        projected_features = self.projection_layer(pooled)  # [batch_size, projection_dim]

        # (Option 1) Classifiers act on the projected features
        logits = torch.cat([
            classifier(projected_features) for classifier in self.classifiers
        ], dim=-1)

        # MODIFIED: Return logits and projected features (h_client_adapted)
        return logits, projected_features


def load_stu_model(args, device, teacher_model):  # Added teacher_model parameter
    # 1. Rebuild student model's Llama base structure
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
    if llama.config.pad_token_id is None:
        llama.config.pad_token_id = tokenizer.pad_token_id
    llama_lora = get_peft_model(llama, lora_config)

    # Teacher LLM's hidden_size as student model's projection target dimension
    teacher_hidden_size = 4096

    # 2. Instantiate student model (Llama4SQA_prompt)
    # At this point, the classification heads are randomly initialized
    student_model = Llama4SQA_prompt_client(
        llama_lora,
        num_labels=9,
        projection_dim=teacher_hidden_size
    ).to(device)
    print("Student model structure has been created.")

    print("Trying to extract classifier head weights directly from the loaded teacher model object...")
    try:
        # Extract directly from the teacher model's state dictionary in memory without any I/O operations
        teacher_state_dict = teacher_model.state_dict()

        classifier_weights = {}
        for key, value in teacher_state_dict.items():
            if key.startswith("LLM_model.classifiers"):
                new_key = key.replace("LLM_model.", "")
                classifier_weights[new_key] = value

        if not classifier_weights:
            raise KeyError("No 'LLM_model.classifiers' weights found in the teacher model object.")

        # Load weights
        student_model.load_state_dict(classifier_weights, strict=False)
        print("Successfully loaded teacher model's classifier head weights into the student model.")
        # (Optional) Freeze the classifier head so it doesn't update during training
        for param in student_model.classifiers.parameters():
            param.requires_grad = False
        print("Student model's classifier head has been frozen and will not update during training.")
    except Exception as e:
        print(f"Failed to extract weights from teacher model object: {e}")
        print("Warning: Student model will use randomly initialized classifier heads.")

    return student_model, tokenizer


def features_to_probs(features, temperature=1.0):
    """Use Softmax to convert feature vectors to probability distributions."""
    return F.softmax(features / temperature, dim=-1)


def compute_jsd(p, q, eps=1e-8):
    """Calculate Jensen-Shannon Divergence between two probability distributions p and q."""
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(torch.log(m + eps), p, reduction='none').sum(-1)
    kl_qm = F.kl_div(torch.log(m + eps), q, reduction='none').sum(-1)
    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd.mean()


def train_distillation_client(args):
    """
    teacher_model: Already trained teacher model (cloud)
    student_model: Student model to be trained (edge)
    """
    device = args.device

    # Load teacher model (for inference)
    teacher_model_path = args.teacher
    teacher_model, _ = load_combined_model(load_path=teacher_model_path, args=args, device=device)

    # 2. Load student model and pass the already loaded teacher_model
    #    So load_stu_model doesn't need to read from disk
    student_model, tokenizer = load_stu_model(args, device, teacher_model=teacher_model)
    # Set mode
    teacher_model.to(device)
    student_model.to(device)
    teacher_model.eval()  # Teacher model inference mode
    # Data
    train_data = CombinedDataset(args, 'train')  # No tokenizer needed
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
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Downstream task loss
    epsilon = 0.1
    alpha = 1
    beta = 0
    gamma = 0
    print(f"Loss weights, alpha:{alpha}, beta:{beta}, gamma:{gamma}")
    # 5. Loss function
    loss_fn = EnhancedVACCLoss(
        epsilon=0.1,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    ).to(args.device)
    # Load text embedding model
    text_model = load_text_model(
        vocab_size=10000,
        embedding_dim=64,
        load_path=args.text,
        device=args.device
    )
    student_model.train()
    # --- Add code to print trainable parameter names here ---
    print("=" * 50)
    print("Trainable parameters in student_model:")
    print("=" * 50)

    # Iterate through all named parameters
    for name, param in student_model.named_parameters():
        # Check if the parameter requires gradient updates (i.e., is trainable)
        if param.requires_grad:
            print(f"Name: {name}, Shape: {param.shape}")
    print("=" * 50)
    # for name1, param1 in distill_module.named_parameters():
    #     # Check if the parameter requires gradient updates (i.e., is trainable)
    #     if param1.requires_grad:
    #         print(f"Name: {name1}, Shape: {param1.shape}")
    print("=" * 50)
    # --- End of print code ---
    for epoch in range(args.epochs):
        student_model.train()
        total_loss = 0
        current_lr = optimizer.param_groups[0]['lr']
        print(f'<<-------- Distill epoch {epoch}/{args.epochs}, lr:{current_lr}-------->>')

        # Assume both data loaders have the same number of batches
        for batch_idx, (batch, batch_student) in enumerate(tqdm(zip(dataloader_train, dataloader_train_stu), leave=True)):
            prompt, labels, opt_labels, graph_data, time_series = batch
            input_ids, att_mask, labels_stu, opt_labels_stu = batch_student
            input_ids, att_mask, labels_stu, opt_labels_stu = input_ids.to(device), att_mask.to(device), labels_stu.to(device), opt_labels_stu.to(device)
            opt_labels = opt_labels.to(device)
            # Print differences only if tensors are not equal
            if not torch.equal(opt_labels, opt_labels_stu):
                mismatch_pos = (opt_labels != opt_labels_stu).nonzero(as_tuple=False)
                print(f"[Batch {batch_idx}] ❌ Not equal, first 5 mismatch positions: {mismatch_pos[:5].tolist()}")
            # Process text data
            text_embeddings = text_to_vector(prompt, text_model, args)
            text_embeddings = text_embeddings.to(args.device)

            labels, opt_labels = labels.to(args.device), opt_labels.to(args.device)

            # Process graph data
            if isinstance(graph_data, dict):
                # Check if manual wrapping is needed
                if "batch" not in graph_data:  # If not already wrapped
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

            # -----------------------------------------------
            # 1) Teacher model inference, get logits and h_server (pooled feature)
            with torch.no_grad():
                teacher_model.eval()
                # teacher_logits is the final output, teacher_features is the pooled feature inside Llama4SQA (h_server)
                teacher_logits, teacher_features = teacher_model(
                    text_inputs=text_embeddings,
                    graph_data=graph_batch,
                    time_series=time_series
                )
                teacher_features = teacher_features.detach()
            # 2) Student model inference, get logits and h_client_adapted
            student_model.train()
            # student_logits is the final output, student_features is the pooled feature after projection (h_client_adapted)
            student_logits, student_features = student_model(input_ids=input_ids, attention_mask=att_mask)

            # Calculate normalized MSE loss
            loss = F.mse_loss(student_features, teacher_features)

            # -----------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_epoch_loss = total_loss / len(dataloader_train)
        print(f'<<-------- Epoch {epoch} done, Avg Loss: {avg_epoch_loss:.4f} -------->>')
        # Validation
        test_len = len(dataloader_test_stu)
        student_model.eval()
        with torch.no_grad():
            all_pred_opt = []
            all_opt_labels = []
            for batch_idx, batch in enumerate(tqdm(dataloader_test_stu, leave=True)):
                input_ids, att_mask, labels, opt_labels = batch
                input_ids, att_mask, labels, opt_labels = input_ids.to(device), att_mask.to(device), labels.to(
                    device), opt_labels.to(device)
                logits, h_client = student_model(input_ids=input_ids, attention_mask=att_mask)
                probs = torch.sigmoid(logits)
                # probs = logits
                pred_opt = probs
                all_pred_opt.append(pred_opt)
                all_opt_labels.append(opt_labels)
                # Merge all batch prediction results and true labels
            all_pred_opt = torch.cat(all_pred_opt, dim=0)
            all_opt_labels = torch.cat(all_opt_labels, dim=0)
            # Calculate regression metrics
            mse = torch.mean((all_pred_opt - all_opt_labels) ** 2).item()
            mae = torch.mean(torch.abs(all_pred_opt - all_opt_labels)).item()
            # Calculate V - ACC
            all_multilabel_pred = (all_pred_opt > 0.5).float()
            all_multilabel_true = (all_opt_labels > 0.1).float()
            right_label_all = torch.where(all_multilabel_true == all_multilabel_pred, 1, 0).sum()
            vacc = right_label_all / test_len / 9
            metrics = calculate_metrics(all_multilabel_pred, all_opt_labels)
            print("\nNew evaluation metrics:")
            print(f"F1 Micro: {metrics['F1_micro']:.4f}")
            print(f"F1 Macro: {metrics['F1_macro']:.4f}")
            print(f"AUC Micro: {metrics['AUC_micro']:.4f}")
            print(f"AUC Macro: {metrics['AUC_macro']:.4f}")
            print(f"Subset Accuracy: {metrics['Subset_Accuracy']:.4f}")
            if vacc > 0.6836 and metrics['AUC_micro'] > 0.5265 and metrics['F1_macro'] > 0.5114:
                # <<<<<<<<<<<<<<< Part to save model weights >>>>>>>>>>>>>>>
                print("Start saving student model's trainable weights...")

                # 1. Get LoRA weights
                lora_weights = get_peft_model_state_dict(student_model.llama)

                # 2. Get weights of our added layers (adapter and classifier)
                # student_model is an instance of Llama4SQA_prompt
                full_student_state_dict = student_model.state_dict()

                additional_weights = {}
                for name, param in full_student_state_dict.items():
                    # Filter out projection_layer and classifiers parameters
                    if name.startswith("projection_layer.") or name.startswith("classifiers."):
                        additional_weights[name] = param

                # 3. Merge all weights to be saved into a dictionary
                # We save them under different keys for clarity when loading later
                final_weights_to_save = {
                    'lora_weights': lora_weights,
                    'projection_weights': {k: v for k, v in additional_weights.items() if
                                           k.startswith("projection_layer.")},
                }

                # 4. Define save path and save
                save_path = f"{args.pre_path}/{args.dataset}_{vacc}distilled_student_model_weights.pth"
                torch.save(final_weights_to_save, save_path)

                print(f"Student model's trainable weights have been successfully saved to: {save_path}")
                print("Saved contents include:")
                print(f" - LoRA weights: {len(lora_weights)} tensors")
                print(f" - Adapter layer weights: {len(final_weights_to_save['projection_weights'])} tensors")
        scheduler.step()
    print('---------------- Distillation Finished ----------------')


def calculate_metrics(all_pred_labels, all_opt_labels, device='cuda:0'):
    """
    Calculate F1, AUC, and Subset Accuracy metrics

    Parameters:
    all_pred_labels (torch.Tensor): Predicted labels
    all_opt_labels (torch.Tensor): True labels
    device (str): Calculation device, default is cuda:0

    Returns:
    dict: Dictionary containing various evaluation metrics
    """
    # Ensure all tensors are on the same device
    all_pred_labels = all_pred_labels.to(device)
    all_opt_labels = all_opt_labels.to(device)

    # Convert to numpy arrays
    y_pred = all_pred_labels.cpu().numpy().flatten()  # Move to CPU first then convert
    y_true = (all_opt_labels > 0.1).float().cpu().numpy().flatten()  # Move to CPU first then convert

    # Calculate F1 score (micro and macro)
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    # Calculate AUC (micro and macro)
    try:
        auc_micro = roc_auc_score(y_true, y_pred, average='micro')
        auc_macro = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        # Handle cases with only one class
        auc_micro = 0.0
        auc_macro = 0.0
        print("Cannot calculate AUC, may have only one class")

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


def load_text_model(vocab_size, embedding_dim, load_path, device):
    """Load saved text_model (nn.Embedding)"""
    text_model = nn.Embedding(vocab_size, embedding_dim).to(device)
    text_model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))
    text_model.eval()  # Use eval mode for inference, switch back to train for training
    print(f"Text embedding model loaded: {load_path}")
    return text_model


def hash_word(word, vocab_size):
    # Use MD5 hash, take first 8 hex digits as integer, ensure result is fixed
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
            print("Using zero-prompt")
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

    # Create tensor and ensure it's on the same device as the model
    indices_tensor = torch.tensor(indices, dtype=torch.long, device=model.weight.device)

    # Use embedding layer to get vectors
    vectors = model(indices_tensor)

    # Calculate average vector
    return torch.mean(vectors, dim=0)


def load_combined_model(load_path, args, device):
    """Load joint model weights from path, rebuild sub-model structures"""
    print(f"Loading joint model weights from {load_path}...")

    # 1. Rebuild LLM model
    model_path = args.pre_path + '/models/LLAMA3.1-8B-Instruct/'
    tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    tokenizer.pad_token = tokenizer.eos_token
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,  # Inference mode
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    llama = AutoModelForCausalLM.from_pretrained(model_path)
    llama_lora = get_peft_model(llama, lora_config)
    llm_model = Llama4SQA_sever(llama_lora, num_labels=9).to(device)

    # 2. Rebuild graph model
    graph_model = PlanGAT(in_channels=11, hidden_channels=128, out_channels=256).to(device)

    # 3. Rebuild time series model
    # time_model = TimeLSTM(input_size=7, hidden_size=64, output_size=256).to(device)
    time_model = TimeLSTM(input_size=1, hidden_size=64, q=7, output_dim=40, num_layers=1)
    # 4. Initialize joint model (structure must be exactly the same as when saved)
    combined_model = CombinedModel_sever(
        LLM_model=llm_model,
        graph_model=graph_model,
        time_model=time_model,
        fusion_type='add',
        text_dim=64,
        graph_dim=256,
        time_dim=280,
        fusion_dim=4096
    ).to(device)

    # Load joint model weights
    try:
        model_state = torch.load(load_path, weights_only=True)
    except:
        model_state = torch.load(load_path, map_location=torch.device('cpu'), weights_only=True)

    combined_model.load_state_dict(model_state['combined_model_state'])
    print(f"Joint model weights successfully loaded from epoch {model_state['epoch']}")

    return combined_model, tokenizer