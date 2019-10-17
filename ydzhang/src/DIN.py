from torch import nn
import torch
import torch.functional as F
from src.attentionSequencePoolingLayer import SequenceAttentionPoolingLayer
from src.layers import FullyConnectedLayer

# TODO:
class DIN(nn.Module):
    def __init__(self,
                 query_dim,
                 hist_behavior_dim,
                 user_profile_dim,
                 hidden_dim_list = [512, 1],
                 use_batchnorm=True,
                 user_sigmoid = True,
                 dropout=0):
        super(DIN, self).__init__()

        self.query_dim = query_dim
        self.hist_behavior_dim = hist_behavior_dim
        self.user_profile_dim = user_profile_dim
        self.hidden_dims = hidden_dim_list
        self.sequenceAttentionPoolingLayer= SequenceAttentionPoolingLayer(self.query_dim, self.hist_behavior_dim)
        self.no_hist_embedding = nn.Parameter(torch.randn((self.hist_behavior_dim, )))
        self.concated_dim = self.query_dim + self.hist_behavior_dim + self.user_profile_dim
        self.output_layer = FullyConnectedLayer(self.concated_dim,
                                                self.hidden_dims,
                                                sigmoid= user_sigmoid,
                                                batch_norm= use_batchnorm,
                                                dropout_rate= dropout)

    def forward(self, query_embedding, hist_embedding, hist_length, user_profile_embedding):

        # add wide part of model ?
        hist_pooled_embedding = self.sequenceAttentionPoolingLayer(query_embedding, hist_embedding, hist_length)
        print('number of zero length history %d' %(torch.sum(hist_length == 0), ))
        hist_pooled_embedding = torch.where(hist_length == 0, hist_pooled_embedding, self.no_hist_embedding) # amazing shape broadcast
        embed_concated = torch.cat([query_embedding, hist_pooled_embedding, user_profile_embedding], dim= -1)


        logits = self.output_layer(embed_concated)

        return logits


if __name__ == '__main__':
    query = torch.randn(3, 200)
    hist_behavior = torch.randn(3, 10, 100)
    hist_length = 8 * torch.ones(3, 1)
    user_profile_embedding = torch.randn(3, 300)

    t = DIN(200, 100, 300)
    out = t(query, hist_behavior, hist_length, user_profile_embedding)
    print(out.detach().numpy())