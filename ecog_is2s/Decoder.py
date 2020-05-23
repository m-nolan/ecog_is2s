import torch
import torch.nn as nn

class Decoder_GRU(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, seq_len, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.seq_len = seq_len

        self.rnn = nn.GRU(output_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout) # no dropout is added to the end of an rnn block in pytorch
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input_data, hidden):
        # input = [batch_size, seq_len, hid_dim]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # hidden = [n layers, batch size, hid dim]

#         input_data = input_data.unsqueeze(0) # not sure if this this is needed for not-embedded inputs
#         breakpoint()
        output, hidden = self.rnn(input_data, hidden)

        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #"seq len and n directions will always be 1 in the decoder, therefore:" <- figure out how to change this
        #output = [batch_size, 1, hid dim]
        #hidden = [n layers, batch size, hid dim]

        prediction = self.fc_out(output)

        return prediction, output, hidden # predicted ECoG signal, decoder states, last decoder state

class Decoder_LSTM(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(hid_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # hidden = [n layers, batch size, hid dim]

        input = input.unsqueeze(0) # not sure if this this is needed for not-embedded inputs

        output, (hidden, cell) = self.rnn(input, (hidden, cell))

        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden, cell
