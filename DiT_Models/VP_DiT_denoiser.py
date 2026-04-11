import torch
import torch.nn as nn
from DiT_model import DiT_NN

'''
This is to implement the VP version of JiT
'''

class VP_DiTDenoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = DiT_NN(img_size=args.img_size,
                           in_channels=args.data_channels,
                           patch_size=args.patch_size,
                           hidden_size=args.hidden_size,
                           num_heads=args.num_heads,
                           depth=args.depth,
                           mlp_ratio=args.mlp_ratio,
                           attn_drop=args.attn_drop,
                           proj_drop=args.proj_drop)

        self.img_size = args.img_size
        self.P_mean = args.P_mean # for sample t
        self.P_std = args.P_std # for sample t

        self.t_eps = args.t_eps # used to clip the denominator when computing velocity
        self.noise_scale = args.noise_scale # the noise scale for noise the clean image

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = {}
        self.ema_params2 = {}

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)


    def forward(self, x):
        # x: [B, C, N_t, N_r]
        # labels_dropped = self.drop_labels(labels) if self.training else labels
        # x = 2 * x - 1

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))  # [B, 1, 1, 1]
        e = torch.randn_like(x) * self.noise_scale  # [B, C, N_t, N_r]

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)  # [B, C, N_t, N_r]

        x_pred = self.net(z, t.flatten())
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss


    @torch.no_grad()
    def update_ema(self):
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # if the first time or has new parameters
            if name not in self.ema_params1:
                self.ema_params1[name] = p.detach().clone()
            else:
                self.ema_params1[name].mul_(self.ema_decay1).add_(p.data, alpha=1 - self.ema_decay1)

            if name not in self.ema_params2:
                self.ema_params2[name] = p.detach().clone()
            else:
                self.ema_params2[name].mul_(self.ema_decay2).add_(p.data, alpha=1 - self.ema_decay2)

    def init_ema(self):
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            self.ema_params1[name] = p.detach().clone()
            self.ema_params2[name] = p.detach().clone()




