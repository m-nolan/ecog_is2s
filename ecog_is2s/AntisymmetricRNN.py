import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

class AntisymmetricRNN(nn.Module):
    def __init__(self, input_dim, output_classes, n_units=32, eps=0.01, gamma=0.01, use_gating=True, init_W_std=1,
                 is_cuda=True):
        super(AntisymmetricRNN, self).__init__()

        normal_sampler_V = torch.distributions.Normal(torch.Tensor([0]) # mean
                                                    , torch.Tensor([1/input_dim])) # variance
        # init Vh
        Vh_init_weight = nn.Parameter(normal_sampler_V.sample((n_units,input_dim))[...,0])
        Vh_init_bias = nn.Parameter(torch.zeros(n_units))
        self.Vh = nn.Linear(input_dim,n_units)
        self.Vh.weight = Vh_init_weight
        self.Vh.bias = Vh_init_bias

        if use_gating:
            # init Vz
            Vz_init_weight = nn.Parameter(normal_sampler_V.sample((n_units,input_dim))[...,0])
            Vz_init_bias = nn.Parameter(torch.zeros(n_units))
            self.Vh = nn.Linear(input_dim,n_units)
            self.Vh.weight = Vz_init_weight
            self.Vh.bias = Vz_init_bias

        # init W - the antisymmetric bit
        normal_sampler_W = torch.distributions.Normal(torch.Tensor([0]),torch.Tensor([init_W_std/input_dim]))
        self.W = nn.Parameter(normal_sampler_W.sample((n_units,input_dim))[...,0])
        # can this be implemented in a more sparse way?

        # init diffusion - scaled I
        self.gamma_I = gamma*torch.eye(n_units, n_units)
        if is_cuda:
            self.gamma_I = self.gamma_I.cuda()

        self.eps = eps
        self.use_gating = use_gating
        self.is_cuda = is_cuda
        self.n_units = n_units
        self.fully_connected = nn.Linear(n_units,output_classes) # note that this could be a continuous output space

    def forward(self, x):
        # x.shape = (num_batch, timesteps, input_dim)
        h = torch.zeros(x.shape[0], self.n_units)
        if self.is_cuda:
            h = h.cuda()
        T = x.shape[1]

        if self.use_gating:
            for t in range(T):
                # (W - WT - gammaI)h
                WmWT_h = torch.matmul(h, (self.W - self.W.transpose(1,0) - self.gamma_I))
                # Vhx + bh
                Vh_x = self.Vh(x[:,t,:]) # look at that - linear layers are functions!
                # (W - WT - gammaI)h + Vhx + bh
                linear_transform_1 = WmWT_h + Vh_x
                # Vzx + bz
                Vz_x = self.Vz(x[:,t,:])
                # (W - WT - gammaI)h + Vxx + bz
                linear_transform_2 = WmWT_h + Vz_x
                # gated activation
                f = torch.tanh(linear_transform_1)*torch.sigmoid(linear_transform_2)
                # ODE-like difference update
                h = h + self.eps * f
        else: # no gating
            for t in range(T):
                # (W - WT - gammaI)h
                WmWT_h = torch.matmul(h, (self.W - self.W.transpose(1,0) - self.gamma_I))
                # Vhx + bh
                Vh_x = self.Vh(x[:,t,:]) # look at that - linear layers are functions!
                # (W - WT - gammaI)h + Vhx + bh
                linear_transform_1 = WmWT_h + Vh_x
                f = torch.tanh(linear_transform_1)
                # ODE-like difference update
                h = h + self.eps * f

        out = self.fully_connected(h)

        return out
