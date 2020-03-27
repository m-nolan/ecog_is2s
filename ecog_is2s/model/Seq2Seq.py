import torch
import torch.nn as nn

class Seq2Seq_GRU(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device # what is this?
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Encoder, decoder embedding dimensions (hidden state) must be equal."
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder, decoder layer number must be equal."
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio: prob. to use teacher forcing
        #e.g. if 0.75, ground-truth imports are used 75% of the time
        
        batch_size = trg.shape[0]
        
        src_len = src.shape[1]
        src_dim = src.shape[2]
        
        trg_len = trg.shape[1]
        trg_dim = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_dim).to(self.device)
        
        enc_state, hidden = self.encoder(src)
        
        output = src[:,-1,:].unsqueeze(1) # start the decoder with the actual output
        
        for t in range(trg_len): # ignore that first data point
            pred, output, hidden = self.decoder(output,hidden)
            
            outputs[:,t,:] = pred.squeeze()
            
            teacher_force = random.random() < teacher_forcing_ratio
            
            input = trg[:,t,:] if teacher_force else output
        
        return outputs
    