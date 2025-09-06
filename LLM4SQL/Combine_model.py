import torch.nn as nn
import torch


class CombinedModel(nn.Module):
    def __init__(self, LLM_model, graph_model, time_model, fusion_type='concat',
                 text_dim=512, graph_dim=64, time_dim=64, fusion_dim=2560):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.graph_proj = nn.Linear(graph_dim, fusion_dim)
        self.time_proj = nn.Linear(time_dim, fusion_dim)
        self.LLM_model = LLM_model
        self.graph_model = graph_model
        self.time_model = time_model
        self.fusion_dim = fusion_dim
        self.fusion_type = fusion_type

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(fusion_dim // 2, fusion_dim)
        )

    def forward(self, text_inputs, graph_data, time_series):

        text_embedding = text_inputs
        text_embedding = text_embedding.to("cuda:0")

        graph_embedding = self.graph_model(graph_data)
        graph_embedding =graph_embedding.to("cuda:0")

        time_embedding = self.time_model(time_series)
        time_embedding = time_embedding.squeeze(0)
        time_embedding = time_embedding.to("cuda:0")

        if self.fusion_type == 'concat':
            fusion_input = torch.cat([text_embedding, graph_embedding, time_embedding], dim=0)
        elif self.fusion_type == 'add':

            text_emb = self.text_proj(text_embedding)
            graph_emb = self.graph_proj(graph_embedding)
            time_emb = self.time_proj(time_embedding)
            fusion_input = text_emb + graph_emb + time_emb
        else:
            raise ValueError("Unsupported Fusion Types")


        fusion_embeding = self.fusion_layer(fusion_input)



        fusion_embeding = fusion_embeding.unsqueeze(0).unsqueeze(1)

        output = self.LLM_model(inputs_embeds=fusion_embeding)
        return output

    def _resize_embedding(self, embedding, target_dim):
        if embedding.shape[0] == target_dim:
            return embedding
        return nn.Linear(embedding.shape[0], target_dim)(embedding)