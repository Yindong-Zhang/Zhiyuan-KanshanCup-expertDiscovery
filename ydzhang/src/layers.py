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


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, hidden_size, bias= True, batch_norm=True, dropout_rate=0.5, activation='relu',
                 sigmoid=False):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_size) >= 1
        self.sigmoid = sigmoid

        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0], bias=bias))

        for i, h in enumerate(hidden_size[:-1]):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size[i]))

            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'prelu':
                layers.append(nn.PReLU())
            else:
                raise NotImplementedError

            layers.append(nn.Dropout(p=dropout_rate))
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1], bias=bias))

        self.fc = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()

        # weight initialization xavier_normal (or glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x)
