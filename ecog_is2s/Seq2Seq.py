import torch
import torch.nn as nn
import numpy as np
# import random

class Seq2Seq_GRU(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, enc_len, dec_len, device, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.enc_len = enc_len # do we need these?
        self.dec_len = dec_len # I don't think we need these
        self.dropout = dropout
        self.device = device

        # create encoder
        # self.encoder.rnn = nn.GRU(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.encoder = Encoder_GRU(self.input_dim, self.hid_dim, self.n_layers, self.enc_len, self.dropout)

        # create decoder
        # self.decoder.rnn = nn.GRU(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        # self.decoder.dropout = nn.Dropout(dropout)
        # self.decoder.fc_out = nn.Linear(hid_dim, input_dim)
        self.decoder = Decoder_GRU(self.input_dim, self.hid_dim, self.n_layers, self.dec_len, self.dropout)

    # full model forward pass
    def forward(self, src, trg, teacher_forcing_ratio = 0.00):
        batch_size = trg.shape[0]
        src_len = src.shape[1]
        src_dim = src.shape[2]
        trg_len = trg.shape[1]
        trg_dim = trg.shape[2]

        # preallocate outputs
        out = torch.zeros(batch_size, trg_len, trg_dim).to(self.device)
        dec_state = torch.zeros(batch_size, trg_len, self.hid_dim).to(self.device)

        enc_state, hidden = self.encoder(src)

        input_ = torch.zeros((batch_size, 1, trg_dim)).to(self.device, non_blocking=True)
        for t in range(trg_len):
            # pred: the output of the linear layer, trained to track the ECoG data.
            # output: the output of the decoder and input to the following fc linear layer.
            # hidden: the hidden state of the decoder at the last time point: [n_layer*n_dir, n_batch, n_ch]
            #       ^ if you want to see each layer's activity at the last time point, use this.
            pred, output, hidden = self.decoder(input_,hidden)
            out[:,t,:] = pred.squeeze(1)
            dec_state[:,t,:] = output.squeeze(1)
            teacher_force = torch.rand(1)[0] < teacher_forcing_ratio
            input_ = trg[:,t,:].unsqueeze(1) if teacher_force else pred

        return out, enc_state, dec_state

    def train_iter(self, iterator, optimizer, criterion, clip = 1.0, teacher_forcing_ratio=0.0):
        self.train()

        epoch_loss = 0
        batch_loss = []

        for idx, batch in enumerate(iterator):
            # ^change this when you update the dataloader to split src/trg
            if np.mod(idx+1,1000) == 0:
                print(idx,len(iterator))
            src = batch[:,:self.enc_len,:]
            trg = batch[:,self.enc_len:self.enc_len+self.dec_len,:self.input_dim]

            optimizer.zero_grad()
            output, _, _ = self(src,trg,teacher_forcing_ratio=teacher_forcing_ratio)

            loss = criterion(output,trg)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(),clip)
            optimizer.step()

            epoch_loss += loss.item()
            batch_loss.append(loss.item())

        return epoch_loss, np.array(batch_loss) # is np use here problematic?

    # pytorch-lightning hooks
    def training_step( step, batch, batch_idx ):
        src, trg = batch
        out, _, _ = self(src, trg)
        loss = nn.functional.mse_loss(trg,out)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def eval_iter(self, iterator, criterion):
        self.eval()

        epoch_loss = 0.
        batch_loss = []

        with torch.no_grad():
            for idx, batch in enumerate(iterator):
                if np.mod(idx+1,1000)==0:
                    print(idx,len(iterator))
                src = batch[:,:self.enc_len,:]
                trg = batch[:,self.enc_len:self.enc_len+self.dec_len,:self.input_dim]

                output, enc_state, dec_state = self(src,trg,teacher_forcing_ratio=0.)

                loss = criterion(output, trg)
                epoch_loss += loss.item()
                batch_loss.append(loss.item())

        return epoch_loss, np.array(batch_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    # for pytorch-lightning integration
    def train_dataloader(self):
        # dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        # loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
        # return loader
        None
        # ending future pytorch-lightning integration

class Encoder_GRU(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, seq_len, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.seq_len = seq_len

        self.rnn = nn.GRU(input_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

    def forward(self, input_data):
        output, hidden = self.rnn(input_data)

        return output, hidden

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
        output, hidden = self.rnn(input_data, hidden)
        prediction = self.fc_out(output)

        return prediction, output, hidden

### Deprecated ###
# old implementation, used separate enc/dec initializations
class _Seq2Seq_GRU_old(nn.Module):
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

        # input_ = src[:,-1,:trg_dim].unsqueeze(1) # start the decoder with the actual output, remove dx if present
        input_ = torch.zeros((batch_size,1,trg_dim)).to(self.device, non_blocking=True)

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
