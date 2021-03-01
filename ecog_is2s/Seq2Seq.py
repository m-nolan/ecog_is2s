import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
# import random

### PYTORCH-LIGHTNING NETWORK IMPLEMENTATION ###
class Seq2seq_ptl(LightningModule):
    # LightningModule class definition for encoder-decoder seq2seq networks.
    def __init__( self, input_dim, hid_dim, n_layers, device='cpu', criterion='MSE', dropout=0.0, learning_rate=0.005,
                  teacher_forcing_ratio=0.0, use_diff=False, bidirectional=False, split=(0.8,0.2,0.0) ):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.device_name = device
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.use_diff = use_diff
        self.bidirectional = bidirectional
        self.split = split # ( train_frac, valid_frac, test_frac )

        # compute corrected component dimensions
        if self.use_diff:
            enc_input_dim = 2*self.input_dim
        else:
            enc_input_dim = self.input_dim

        if self.bidirectional:
            dec_hidden_dim = 2*self.hid_dim
        else:
            dec_hidden_dim = self.hid_dim

        # create network components
        self.encoder = Encoder_GRU_ptl(enc_input_dim,self.hid_dim,self.n_layers,self.dropout,self.bidirectional)
        self.decoder = Decoder_GRU_ptl(self.input_dim,dec_hidden_dim,self.n_layers,self.dropout)

        # initialize network weights: use default initialization for now (uniform random draws, ±sqrt(1/n_hid))

        # configure loss criterion
        self.configure_criterion(criterion)

    # forward pass - compute network outputs
    def forward(self, src, trg):
        batch_size = trg.shape[0]
        src_len = src.shape[1]
        src_dim = src.shape[2]
        trg_len = trg.shape[1]
        trg_dim = trg.shape[2]

        # preallocate outputs
        out = torch.zeros(batch_size, trg_len, trg_dim)
        if self.bidirectional:
            dec_dim = 2*self.hid_dim
        else:
            dec_dim = self.hid_dim
        dec_state = torch.zeros(batch_size, trg_len, dec_dim)
        enc_state, hidden = self.encoder(src)
        if self.bidirectional:
            # unwrap bidirectional pass output shapes
            # torch.reshape() didn't have a convenient way to do this. Reassess?
            hidden = torch.cat((hidden[:self.n_layers,],hidden[self.n_layers:,]),axis=-1)

        # change initialization!
        # input_ = torch.zeros((batch_size, 1, trg_dim)).to(self.device, non_blocking=True)
        input_ = src[:,-1,:self.input_dim].unsqueeze(axis=1).to(self.device, non_blocking=True)
        for t in range(trg_len):
            # pred: the output of the linear layer, trained to track the ECoG data.
            # output: the output of the decoder and input to the following fc linear layer.
            # hidden: the hidden state of the decoder at the last time point: [n_layer*n_dir, n_batch, n_ch]
            #       ^ if you want to see each layer's activity at the last time point, use this.
            pred, output, hidden = self.decoder(input_,hidden)
            out[:,t,:] = pred.squeeze(1)
            dec_state[:,t,:] = output.squeeze(1)
            teacher_force = torch.rand(1)[0] < self.teacher_forcing_ratio
            input_ = trg[:,t,:].unsqueeze(1) if teacher_force else pred

        return out, enc_state, dec_state

    # configure loss criterion
    def configure_criterion( self, criterion ):
        if criterion == 'mse' or criterion == 'MSE' or criterion == 'L2':
            self.criterion = nn.MSELoss()
        elif criterion == 'lasso' or criterion == 'LASSO' or criterion == 'L1':
            self.criterion = nn.L1Loss()
        elif type(criterion).__bases__[0].__name__ == '_Loss':
            self.criterion = criterion

    def prepare_data( self ):
        self.dataset = None

    # configure parameter optimization algorithm (ADAM)
    def configure_optimizers( self ):
        return torch.optim.Adam(self.parameters(),lr=self.learning_rate)

    # dataloader configuration
    def train_dataloader( self ):
        return None

    def val_dataloader( self ):
        return None

    def test_dataloader( self ):
        return None

    # training loop methods
    def training_step( self, batch, batch_idx ):
        src, trg = batch
        out, enc, dec = self(src,trg)
        loss = self.criterion(out,trg)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    # validation loop methods
    def validation_step( self, batch, batch_idx ):
        src, idx = batch
        out, enc, dec = self(src,trg)
        loss = self.criterion(out,trg)
        return {'val_loss': loss}

    def validation_epoch_end( self, outputs ):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    # test loop methods
    def test_step( self, batch, batch_idx ):
        src, trg = batch
        out, enc, dec = self(src,trg)
        loss = self.criterion(out,trg)
        return {'test_loss': loss} # can I add FVE to this?

    def test_epoch_end( self, outputs ):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

class Encoder_GRU_ptl(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout, bidirectional=False):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(self.input_dim, self.hid_dim, self.n_layers, dropout=dropout,
                          batch_first=True, bidirectional=self.bidirectional)

    def forward(self, input_data):
        output, hidden = self.rnn(input_data)
        return output, hidden

class Decoder_GRU_ptl(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, dropout, bidirectional=False):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(self.output_dim, self.hid_dim, self.n_layers, dropout=dropout,
                          batch_first=True, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(dropout) # no dropout is added to the end of an rnn block in pytorch
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input_data, hidden):
        output, hidden = self.rnn(input_data, hidden)
        prediction = self.fc_out(output)
        return prediction, output, hidden

### ~~~ ###

### non-PT-L network code ###
class Seq2Seq_GRU(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, enc_len, dec_len, device,
                 dropout=0.0, use_diff=False, bidirectional=False):
        super().__init__()
        self.use_diff = use_diff
        self.bidirectional = bidirectional
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.enc_len = enc_len # do we need these?
        self.dec_len = dec_len # I don't think we need these
        self.dropout = dropout
        self.device = device

        # create encoder
        if self.use_diff:
            enc_in_dim = 2*self.input_dim
        else:
            enc_in_dim = self.input_dim
        self.encoder = Encoder_GRU(enc_in_dim, self.hid_dim, self.n_layers,
                                   self.enc_len, self.dropout, bidirectional=self.bidirectional)

        # create decoder
        if self.bidirectional: # account for bidirectional concatenation
            dec_hid_dim = 2*self.hid_dim
        else:
            dec_hid_dim = self.hid_dim
        self.decoder = Decoder_GRU(self.input_dim, dec_hid_dim, self.n_layers,
                                   self.dec_len, self.dropout)

    # full model forward pass
    # @torch.jit.script # this may make things faster, so long as they work at all...
    def forward(self, src, trg, teacher_forcing_ratio = 0.00):
        batch_size = trg.shape[0]
        src_len = src.shape[1]
        src_dim = src.shape[2]
        trg_len = trg.shape[1]
        trg_dim = trg.shape[2]

        # preallocate outputs
        out = torch.zeros(batch_size, trg_len, trg_dim).to(self.device)
        if self.bidirectional:
            dec_dim = 2*self.hid_dim
        else:
            dec_dim = self.hid_dim
        dec_state = torch.zeros(batch_size, trg_len, dec_dim).to(self.device)
        enc_state, hidden = self.encoder(src)
        if self.bidirectional:
            # unwrap bidirectional pass output shapes
            # torch.reshape() didn't have a convenient way to do this. Reassess?
            hidden = torch.cat((hidden[:self.n_layers,],hidden[self.n_layers:,]),axis=-1)

        # change initialization!
        # input_ = torch.zeros((batch_size, 1, trg_dim)).to(self.device, non_blocking=True)
        input_ = src[:,-1,:self.input_dim].unsqueeze(axis=1).to(self.device, non_blocking=True)
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

        for idx, (src,trg) in enumerate(iterator):
            # ^change this when you update the dataloader to split src/trg
            if np.mod(idx+1,1000) == 0:
                print(idx,len(iterator))
            # src = batch[:,:self.enc_len,:]
            # trg = batch[:,self.enc_len:self.enc_len+self.dec_len,:self.input_dim]

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
            for idx, (src,trg) in enumerate(iterator):
                if np.mod(idx+1,1000)==0:
                    print(idx,len(iterator))
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
    def __init__(self, input_dim, hid_dim, n_layers, seq_len, dropout, bidirectional=False):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(self.input_dim, self.hid_dim, self.n_layers, dropout=dropout,
                          batch_first=True, bidirectional=self.bidirectional)

    def forward(self, input_data):
        output, hidden = self.rnn(input_data)

        return output, hidden

class Decoder_GRU(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, seq_len, dropout, bidirectional=False):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(self.output_dim, self.hid_dim, self.n_layers, dropout=dropout,
                          batch_first=True, bidirectional=self.bidirectional)
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
