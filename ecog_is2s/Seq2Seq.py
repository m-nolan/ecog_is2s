import torch
import torch.nn as nn
import random

class Seq2Seq_GRU(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Encoder, decoder embedding dimensions (hidden state) must be equal."
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder, decoder layer number must be equal."

    def forward(self, src, trg, teacher_forcing_ratio = 0.05):

        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio: prob. to use teacher forcing
        #e.g. if 0.75, ground-truth imports are used 75% of the time

        batch_size = trg.shape[0]

        src_len = src.shape[1]
        src_dim = src.shape[2]

        trg_len = trg.shape[1]
        trg_dim = self.decoder.output_dim

        hid_dim = self.encoder.hid_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_dim).to(self.device)
        dec_state = torch.zeros(batch_size, trg_len, hid_dim).to(self.device)

        enc_state, hidden = self.encoder(src)

        input_ = src[:,-1,:trg_dim].unsqueeze(1) # start the decoder with the actual output, remove dx if present

        for t in range(trg_len): # ignore that first data point
            # pred: the output of the linear layer, trained to track the ECoG data.
            # output: the output of the decoder and input to the following fc linear layer.
            # hidden: the hidden state of the decoder at the last time point: [n_layer*n_dir, n_batch, n_ch]
            #       ^ if you want to see each layer's activity at the last time point, use this.
            pred, output, hidden = self.decoder(input_,hidden)
            outputs[:,t,:] = pred.squeeze(1)
            dec_state[:,t,:] = output.squeeze(1)
            teacher_force = random.random() < teacher_forcing_ratio
            input_ = trg[:,t,:].unsqueeze(1) if teacher_force else pred # the next input is the current predicted value.

        return outputs, enc_state, dec_state
