import torch.nn as nn
import torch


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


class Llama4SQA(nn.Module):
    def __init__(self, llama_lora_model, num_labels=9):
        super().__init__()
        self.llama = llama_lora_model  # This is the model with LoRA injection.
        # self.classifier = nn.Linear(self.llama.config.hidden_size, num_labels)
        # Change 1: Use ModuleList to create independent prediction heads for each label.
        self.classifiers = nn.ModuleList([
            nn.Linear(self.llama.config.hidden_size, 1) for _ in range(num_labels)
        ])
        self.attentionPool = AttentionPool(self.llama.config.hidden_size)
        # Initialize classifier weights
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        for classifier in self.classifiers:
            # Initialize the weights of the classifier's linear layer using Xavier Normal
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

        # Mean Pooling without attention mask
        # Directly calculate the mean across all tokens assuming no padding in the input sequence
        pooled = last_hidden.mean(dim=1)  # [batch_size, hidden_size]

        # Change: Predict each label independently
        logits = []
        for classifier in self.classifiers:
            logit = classifier(pooled)  # [batch, 1]
            logits.append(logit)

        # Concatenate predictions for all labels
        logits = torch.cat(logits, dim=1)  # [batch, num_labels]
        return logits