from torch import nn
import torch
import torch.nn.functional as F


class BiLSTMRCNN(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_dim,
                 word_embeddings,
                 hidden_layers = 1,
                 embedding_dropout= 0.1,
                 rnn_dropout = 0.1,
                 ):
        super(BiLSTMRCNN, self).__init__()

        self.embed_size = embedding_size
        self.hidden_num = hidden_layers
        self.hidden_size = hidden_dim
        self.hidden_dropout = rnn_dropout
        self.embedding_dropout_rate = embedding_dropout
        self.bidirectional = True

        self.embedding_layer = nn.Embedding(len(word_embeddings), self.embed_size)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(word_embeddings))
        self.embedding_layer.weight.requires_grad = False
        self.embedding_dropout_layer = nn.Dropout(self.embedding_dropout_rate)

        self.lstm = nn.LSTM(
            self.embed_size,
            self.hidden_size,
            dropout=self.hidden_dropout,
            num_layers=self.hidden_num,
            batch_first=True,
            bidirectional= True,
        )

    def forward(self, input):
        out = self.embedding_layer(input)
        out = self.embedding_dropout_layer(out)
        out, _ = self.lstm(out)

        out = torch.transpose(out, 1, 2)

        out = torch.tanh(out)

        out = F.max_pool1d(out, out.size(2), )

        out = out.squeeze(2)

        return out


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


if __name__ == '__main__':
    import numpy as np
    seq = torch.randint(0, 256, (3, 32))
    word_embeddings = np.random.randn(256, 128)
    t = BiLSTMRCNN(embedding_size= 128, hidden_dim= 256, hidden_layers= 3, word_embeddings= word_embeddings)
    out = t(seq)
    print(out.detach().numpy())
