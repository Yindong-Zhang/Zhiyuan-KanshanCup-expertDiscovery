from torch import nn
import torch
import torch.functional as F
from src.AttentionSequencePoolingLayer import SequenceAttentionPoolingLayer
from src.layers import FullyConnectedLayer

# TODO:
class DIN(nn.Module):
    def __init__(self,
                 query_dim,
                 hist_behavior_dim,
                 user_profile_dim,
                 hidden_dim_list = [512, 1],
                 use_batchnorm=True,
                 dropout=0):
        super(DIN, self).__init__()

        self.query_dim = query_dim
        self.hist_behavior_dim = hist_behavior_dim
        self.user_profile_dim = user_profile_dim
        self.hidden_dims = hidden_dim_list
        feature_embeddings = torch.empty(self.feature_size + 1, self.embedding_size,
                                         dtype=torch.float32, device=self.device,
                                         requires_grad=True)
        nn.init.normal_(feature_embeddings)
        self.feature_embeddings = nn.Parameter(feature_embeddings)
        self.sequenceAttentionPoolingLayer= SequenceAttentionPoolingLayer(self.query_dim, self.hist_behavior_dim)

        self.concated_dim = self.query_dim + self.hist_behavior_dim + self.user_profile_dim
        self.output_layer = FullyConnectedLayer(self.concated_dim,
                                                self.hidden_dims,
                                                sigmoid= True,
                                                batch_norm= use_batchnorm,
                                                dropout_rate= dropout)

    def forward(self, query_embedding, hist_embedding, hist_length, user_profile_embedding):
        hist_pooled_embedding = self.sequenceAttentionPoolingLayer(query_embedding, hist_embedding, hist_length)

        embed_concated = torch.cat([query_embedding, hist_pooled_embedding, user_profile_embedding])


        logits = self.output_layer(embed_concated)

        return logits