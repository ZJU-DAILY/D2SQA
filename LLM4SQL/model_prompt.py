import torch.nn as nn
import torch
from peft import LoraConfig, get_peft_model, TaskType

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


class Llama4SQA_prompt(nn.Module):
    def __init__(self, llama_lora_model, num_labels=9, hidden_size=None):
        super().__init__()
        # If hidden_size is not specified, try to retrieve it from the model configuration
        if hidden_size is None:
            if hasattr(llama_lora_model, 'config'):
                hidden_size = llama_lora_model.config.hidden_size
            elif hasattr(llama_lora_model, 'base_model') and hasattr(llama.base_model, 'config'):
                hidden_size = llama_lora_model.base_model.config.hidden_size
            else:
                # Default value
                hidden_size = 2048

        self.hidden_size = hidden_size
        self.llama = llama_lora_model  # This is the model with LoRA injection
        # self.classifier = nn.Linear(self.llama.config.hidden_size, num_labels)
        # Change 1: Use ModuleList to create independent prediction heads for each label
        self.classifiers = nn.ModuleList([
            nn.Linear(self.hidden_size, 1) for _ in range(num_labels)
        ])
        self.attentionPool = AttentionPool(self.llama.config.hidden_size)
        # Initialize classifier weights
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        for classifier in self.classifiers:
            # Initialize the weights of the linear layer for the classification head using Xavier Normal
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

        last_hidden = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]

        # Mean Pool
        attention_mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        # Set padding positions to 0 and others to 1
        masked_hidden = last_hidden * attention_mask  # Broadcast multiplication to mask out padding
        # Sum and then normalize by the number of valid tokens for each sample to get the mean
        sum_hidden = masked_hidden.sum(dim=1)  # [batch_size, hidden_size]
        valid_token_counts = attention_mask.sum(dim=1)  # [batch_size, 1]
        pooled = sum_hidden / valid_token_counts  # [batch_size, hidden_size]

        # pooled = self.attentionPool(last_hidden)
        # logits = self.classifier(pooled)  # [batch, num_labels]

        # Change 2: Predict each label independently
        logits = []
        for classifier in self.classifiers:
            logit = classifier(pooled)  # [batch, 1]
            logits.append(logit)

        # Concatenate the prediction results for all labels
        logits = torch.cat(logits, dim=1)  # [batch, num_labels]
        return logits