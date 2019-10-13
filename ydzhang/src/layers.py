from torch import nn
import torch
import torch.functional as F


class BiLSTMRCNN(nn.Module):
    def __init__(self, word_embeddings):
        super(BiLSTMRCNN, self).__init__()

        self.embed_size = 200
        self.label_num = 10
        self.embed_dropout = 0.1
        self.fc_dropout = 0.1
        self.hidden_num = 2
        self.hidden_size = 50
        self.hidden_dropout = 0
        self.bidirectional = True

        self.embeddings = nn.Embedding(len(word_embeddings),self.embed_size)
        self.embeddings.weight.data.copy_(torch.from_numpy(word_embeddings))
        self.embeddings.weight.requires_grad = False

        self.lstm = nn.LSTM(
            self.embed_size,
            self.hidden_size,
            dropout=self.hidden_dropout,
            num_layers=self.hidden_num,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        self.embed_dropout = nn.Dropout(self.embed_dropout)
        self.fc_dropout = nn.Dropout(self.fc_dropout)
        self.linear1 = nn.Linear(self.hidden_size * 2, self.hidden_size // 2)
        self.linear2 = nn.Linear(self.hidden_size // 2, self.label_num)

    def forward(self, input):
        out = self.embeddings(input)
        out = self.embed_dropout(out)
        out, _ = self.lstm(out)

        out = torch.transpose(out, 1, 2)

        out = torch.tanh(out)

        out = F.max_pool1d(out, out.size(2))

        out = out.squeeze(2)

        out = self.fc_dropout(out)
        out = self.linear1(F.relu(out))
        output = self.linear2(F.relu(out))

        return output


class MLP(nn.Module):
    def __init__(self, params, use_batchnorm=True, use_dropout=True):
        super(MLP, self).__init__()

        self.embedding_size = params['embedding_size']
        self.field_size = params['field_size']
        self.hidden_dims = params['hidden_dims']
        self.device = params['device']
        self.p = params['p']
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        self.input_dim = self.field_size * self.embedding_size
        self.num_layers = len(self.hidden_dims)

        ## deep weights
        self.deep_layers = nn.Sequential()

        net_dims = [self.input_dim] + self.hidden_dims
        for i in range(self.num_layers):
            self.deep_layers.add_module('fc%d' % (i + 1), nn.Linear(net_dims[i], net_dims[i + 1]).to(self.device))
            if self.use_batchnorm:
                self.deep_layers.add_module('bn%d' % (i + 1), nn.BatchNorm1d(net_dims[i + 1]).to(self.device))
            self.deep_layers.add_module('relu%d' % (i + 1), nn.ReLU().to(self.device))
            if self.use_dropout:
                self.deep_layers.add_module('dropout%d' % (i + 1), nn.Dropout(self.p).to(self.device))

    def forward(self, embeddings):
        deepInput = embeddings.reshape(embeddings.shape[0], self.input_dim)
        deepOut = self.deep_layers(deepInput)
        return deepOut

