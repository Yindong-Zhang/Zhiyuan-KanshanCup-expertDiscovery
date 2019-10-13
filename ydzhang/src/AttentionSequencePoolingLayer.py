
import torch.nn as nn
import torch
import torch.nn.functional as F



class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, embedding_dim=4):
        super(AttentionSequencePoolingLayer, self).__init__()
        # TODO: attention weight normalization
        self.local_att = LocalActivationUnit(hidden_size=[64, 16], bias=[True, True], embedding_dim=embedding_dim, batch_norm=False)


    def forward(self, query_ad, hist_behavior, hist_behavior_length):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        # user behavior length: size -> batch_size * 1
        # output              : size -> batch_size * 1 * embedding_size

        attention_logits = self.local_att(query_ad, hist_behavior)
        attention_logits = torch.transpose(attention_logits, 1, 2)  # B * 1 * T
        # print(attention_score.size())

        # define mask by length
        # user_behavior_length = user_behavior_length.type(torch.LongTensor)
        mask = torch.arange(hist_behavior.size(1))[None, :] < hist_behavior_length[:, None]

        # mask
        output = attention_logits.masked_fill(mask, 1e-9)  # batch_size *

        attention_score = F.softmax(output, dim= -1)
        # multiply weight
        output = torch.matmul(attention_score, hist_behavior)

        return output


class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_size=[80, 40], bias=[True, True], embedding_dim=4, batch_norm=False):
        super(LocalActivationUnit, self).__init__()
        # TODO:
        self.fc1 = FullyConnectedLayer(input_size= 4 *embedding_dim,
                                       hidden_size=hidden_size,
                                       bias=bias,
                                       batch_norm=batch_norm,
                                       activation='dice',
                                       dice_dim=3)



    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size

        user_behavior_len = user_behavior.size(1)
        queries = torch.cat([query for _ in range(user_behavior_len)], dim=1)

        attention_input = torch.cat([queries, user_behavior, queries -user_behavior, queries *user_behavior], dim=-1)
        attention_output = self.fc1(attention_input)

        return attention_output

#     TODO:
class BiInteractionActivationUnit(nn.Module):
    def __init__(self,
                 query_dim,
                 hist_behavior_dim):
        pass
