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


