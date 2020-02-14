import torch
import torch.nn as nn

class Encoder_GRU(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
#         self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(input_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]

        outputs, hidden = self.rnn(src)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        return outputs, hidden

class Encoder_LSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]

        outputs, (hidden, cell) = self.rnn(src)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        return outputs, hidden, cell
