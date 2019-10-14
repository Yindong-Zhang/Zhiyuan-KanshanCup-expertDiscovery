
import torch.nn as nn
import torch
import torch.nn.functional as F
from src.layers import FullyConnectedLayer


class SequenceAttentionPoolingLayer(nn.Module):
    def __init__(self,
                 query_dim,
                 user_hist_dim,
                 ):
        super(SequenceAttentionPoolingLayer, self).__init__()
        self.query_dim = query_dim
        self.user_hist_dim = user_hist_dim
        # self.local_att = LocalActivationUnit(query_dim, user_hist_dim, hidden_size=[64, 16], bias=[True, True], batch_norm=False)
        self.local_att = BiInteractionActivationUnit(query_dim, user_hist_dim)

    def forward(self, query_ad, hist_behavior, hist_behavior_length):
        # query ad            : size -> batch_size * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        # user behavior length: size -> batch_size
        # output              : size -> batch_size * 1 * embedding_size

        attention_logits = self.local_att(query_ad, hist_behavior)
        attention_logits = torch.transpose(attention_logits, 1, 2)  # B * 1 * T
        # print(attention_score.size())

        # define mask by length
        hist_behavior_length = hist_behavior_length.type(torch.LongTensor)
        mask = torch.arange(hist_behavior.size(1))[None, :] < hist_behavior_length[:, None]

        # mask
        output = attention_logits.masked_fill(mask, 1e-9)  # batch_size

        attention_score = F.softmax(output, dim= -1)
        # multiply weight
        output = torch.matmul(attention_score, hist_behavior) # shape: b, 1, d_h

        return output.squeeze(1) # shape: b, d_h


class LocalActivationUnit(nn.Module):
    def __init__(self, query_dim, hist_behavior_dim, hidden_size=[80, 40], bias=[True, True],  batch_norm=False):
        super(LocalActivationUnit, self).__init__()
        self.query_dim = query_dim
        self.hist_behaivor_dim = hist_behavior_dim
        self.input_dim = query_dim + hist_behavior_dim
        self.fc1 = FullyConnectedLayer(input_size= self.input_dim,
                                       hidden_size=hidden_size,
                                       bias=bias,
                                       batch_norm=batch_norm,
                                       activation='prelu')



    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * query_dim
        # user behavior       : size -> batch_size * time_seq_len * hist_behavior_dim
        # ouput               : batchsize, time_seq_len
        user_behavior_len = user_behavior.size(1)
        query = query.unsqueeze(1)
        queries = query.expand(-1, user_behavior_len, -1)

        attention_input = torch.cat([queries, user_behavior], dim=-1)
        attention_output = self.fc1(attention_input)
        return attention_output

class BiInteractionActivationUnit(nn.Module):
    def __init__(self,
                 query_dim,
                 hist_behavior_dim):
        super(BiInteractionActivationUnit, self).__init__()
        self.query_dim = query_dim
        self.hist_behavior_dim = hist_behavior_dim
        self.biInt_weight = nn.Parameter(torch.Tensor(query_dim, hist_behavior_dim))
        nn.init.xavier_normal_(self.biInt_weight)


    def forward(self, query, hist_behavior):
        # TODO:
        attention_logits = torch.einsum('bq, qh, blh->bl', query, self.biInt_weight, hist_behavior)
        return attention_logits[..., None] # shape -> b, l, 1

    def extra_repr(self):
        return "BiInteractionAttentionUnit: %d * %d -> 1" %(self.query_dim, self.hist_behavior_dim)

if __name__ == '__main__':
    query = torch.randn(3, 200)
    hist_behavior = torch.randn(3, 10, 100)
    hist_length = 8 * torch.ones(3, 1)

    a = SequenceAttentionPoolingLayer(200, 100)
    out = a(query, hist_behavior, hist_length).detach().numpy()
    print(out)
