import torch
import torch.nn as nn

# encoder class
class Encoder_GRU(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, seq_len, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.seq_len = seq_len
        
        # gated recurrent layer, dropout layer
        self.rnn = nn.GRU(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        # note: batch_first only permutes dimension order in input and output tensors. It does not affect hidden state.
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_data):
        # input_data: [batch_size x seq_len x input_dim]
        # h0: [n_layers x batch_size x hid_dim]
        batch_size = input_data.size(0)
#         hidden = torch.randn(self.n_layers, batch_size, self.hid_dim) # initialize hidden layer value
        output, hidden = self.rnn(input_data) # hidden initialized as zero tensor
            
        # output = [batch_size x seq_len x hid_dim]
        # hidden = [n layers * n directions, batch size, hid dim]

        return output, hidden

class Encoder_LSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # gated recurrent layer, dropout layer
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_data):
        # src = [src len, batch size]

        outputs, (hidden, cell) = self.rnn(input_data)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        return outputs, hidden, cell
