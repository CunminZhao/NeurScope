import math
import torch
import torch.nn as nn


class ModulationNetwork(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.mod_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.mod_layers.append(
                nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim * 2)
                )
            )
        self._init_weights()

    def _init_weights(self):
        for mod in self.mod_layers:
            nn.init.zeros_(mod[-1].weight)
            nn.init.zeros_(mod[-1].bias)
            with torch.no_grad():
                mod[-1].bias[:mod[-1].bias.shape[0] // 2].fill_(1.0)

    def forward(self, z, layer_idx):
        out = self.mod_layers[layer_idx](z)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma, beta


class SIREN_FiLM(nn.Module):

    def __init__(self, in_ch=4, netD=8, netW=256, out_ch=11,
                 latent_dim=92, w0=30.0, first_w0=30.0, skips=None):
        super().__init__()
        self.w0 = w0
        self.first_w0 = first_w0
        self.netD = netD
        self.skips = skips or []
        self.out_ch = out_ch

        self.out_M = 1
        self.out_L = out_ch - self.out_M
        assert self.out_L >= 1, f"out_ch ({out_ch}) error"

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_ch, netW))
        for i in range(1, netD):
            if i in self.skips:
                self.layers.append(nn.Linear(netW + in_ch, netW))
            else:
                self.layers.append(nn.Linear(netW, netW))

        self.head_M = nn.Linear(netW, self.out_M)   
        self.head_L = nn.Linear(netW, self.out_L)   

        self.modulation = ModulationNetwork(latent_dim, netW, n_layers=netD)

        self._init_siren_weights()

    def _init_siren_weights(self):
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                n = layer.in_features
                if i == 0:
                    layer.weight.uniform_(-1.0 / n, 1.0 / n)
                else:
                    bound = math.sqrt(6.0 / n) / self.w0
                    layer.weight.uniform_(-bound, bound)
                layer.bias.zero_()

            n_M = self.head_M.in_features
            bound_M = math.sqrt(6.0 / n_M) / self.w0
            self.head_M.weight.uniform_(-bound_M, bound_M)
            self.head_M.bias.zero_()

            n_L = self.head_L.in_features
            bound_L = math.sqrt(6.0 / n_L) / self.w0
            self.head_L.weight.uniform_(-bound_L, bound_L)
            self.head_L.weight.mul_(0.01)
            self.head_L.bias.zero_()

    def forward(self, coords, lz):

        input_coords = coords
        h = coords

        for i in range(self.netD):
            if i in self.skips:
                h = torch.cat([input_coords, h], dim=-1)

            h = self.layers[i](h)

            gamma, beta = self.modulation(lz, i)
            h = gamma * h + beta

            current_w0 = self.first_w0 if i == 0 else self.w0
            h = torch.sin(current_w0 * h)


        intensity = self.head_M(h)   
        L_flat    = self.head_L(h)   

        out = torch.cat([intensity, L_flat], dim=-1)
        return out


