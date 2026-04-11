import torch
import torch.nn as nn
import math
from DiT_model import DiT_NN

'''
This is to implement the VE version of SF-DiT-CE
'''

class VE_DiTDenoiser(nn.Module):
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
        self.sigma_data = args.sigma_data

        self.P_mean = args.P_mean
        self.P_std = args.P_std

        self.sigma_min = args.sigma_min
        self.sigma_max = args.sigma_max
        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = {}
        self.ema_params2 = {}

    def get_scaling_for_boundary_condition(self, sigma, sigma_data=1.0, sigma_min=0.0):
        c_skip = (sigma_data ** 2) / ((sigma - sigma_min) ** 2 + sigma_data ** 2)
        c_out = (sigma - sigma_min) * sigma_data / torch.sqrt(sigma ** 2 + sigma_data ** 2)
        c_in = 1 / torch.sqrt(sigma ** 2 + sigma_data ** 2)

        return c_skip, c_out, c_in

    def sample_sigma(self, n: int, device=None):
        rnd = torch.randn(n, device=device)
        log_sigma = (rnd * self.P_std + self.P_mean).clamp(math.log(self.sigma_min), math.log(self.sigma_max))
        sigma = log_sigma.exp()
        return log_sigma, sigma


    def forward(self, x):
        # This is the used configuration of our work, use x-prediction + v-loss
        # x: [B, C, N_t, N_r]
        log_sigma, sigma = self.sample_sigma(x.size(0), device=x.device) # [B]
        sigma = sigma.view(-1, *([1] * (x.ndim - 1)))  # [B, 1, 1, 1]
        eps = torch.randn_like(x)
        z = x + sigma * eps #
        c_skip, c_out, c_in = self.get_scaling_for_boundary_condition(sigma=sigma, sigma_data=self.sigma_data, sigma_min=0.0)
        z_in = z * c_in # multiply c_in before feeding into the net so the net's input is roughly unit scale.
        c_noise = 0.25 * log_sigma # [B]
        x_pred = self.net(z_in, c_noise)
        x_pred = c_skip * z + c_out * x_pred
        v_pred = (x_pred - z) / sigma
        loss = (v_pred + eps) ** 2  # the velocity = -eps
        loss = loss.mean(dim=(1, 2, 3)).mean()
        return loss

    def forward_score_loss(self, x):
        log_sigma, sigma = self.sample_sigma(x.size(0), device=x.device)  # [B]
        sigma = sigma.view(-1, *([1] * (x.ndim - 1)))  # [B, 1, 1, 1]
        eps = torch.randn_like(x)
        z = x + sigma * eps  #
        c_skip, c_out, c_in = self.get_scaling_for_boundary_condition(sigma=sigma, sigma_data=self.sigma_data, sigma_min=0.0)
        z_in = z * c_in
        c_noise = 0.25 * log_sigma  # [B]
        x_pred = self.net(z_in, c_noise)
        x_pred = c_skip * z + c_out * x_pred
        # --------------- score-loss: ----------------
        epsilon_true = (z - x) / sigma
        score_true = -epsilon_true / sigma
        epsilon_pred = (z - x_pred) / sigma
        score_pred = -epsilon_pred / sigma
        loss = (score_true - score_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss


    def forward_x_loss(self, x):
        log_sigma, sigma = self.sample_sigma(x.size(0), device=x.device)  # [B]
        sigma = sigma.view(-1, *([1] * (x.ndim - 1)))  # [B, 1, 1, 1]
        eps = torch.randn_like(x)
        z = x + sigma * eps  #
        c_skip, c_out, c_in = self.get_scaling_for_boundary_condition(sigma=sigma, sigma_data=self.sigma_data,
                                                                      sigma_min=0.0)
        z_in = z * c_in
        c_noise = 0.25 * log_sigma  # [B]
        x_pred = self.net(z_in, c_noise)
        x_pred = c_skip * z + c_out * x_pred
        # --------- x-loss: --------------
        loss = (x_pred - x) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()
        return loss

    def forward_score(self, x):
        log_sigma, sigma = self.sample_sigma(x.size(0), device=x.device)  # [B]
        sigma = sigma.view(-1, *([1] * (x.ndim - 1)))  # [B, 1, 1, 1]
        eps = torch.randn_like(x)
        z = x + sigma * eps  #
        c_skip, c_out, c_in = self.get_scaling_for_boundary_condition(sigma=sigma, sigma_data=self.sigma_data,
                                                                      sigma_min=0.0)
        z_in = z * c_in
        c_noise = 0.25 * log_sigma  # [B]
        # ---- ===== net predicts SCORE directly  ======== ----
        score_pred = self.net(z_in, c_noise)
        score_target = (x - z) / (sigma ** 2)
        loss = (score_pred - score_target) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()
        return loss


    def forward_velocity(self, x):
        log_sigma, sigma = self.sample_sigma(x.size(0), device=x.device)  # [B]
        sigma = sigma.view(-1, *([1] * (x.ndim - 1)))  # [B, 1, 1, 1]
        eps = torch.randn_like(x)
        z = x + sigma * eps
        c_skip, c_out, c_in = self.get_scaling_for_boundary_condition(sigma=sigma, sigma_data=self.sigma_data,
                                                                      sigma_min=0.0)
        z_in = z * c_in
        c_noise = 0.25 * log_sigma  # [B]
        # ---- net predicts VELOCITY directly ----
        v_pred = self.net(z_in, c_noise)
        v_target = (z - x) / sigma  # == eps
        loss = (v_pred - v_target) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()
        return loss

    @torch.no_grad()
    def x0_from_score(self, z, sigma):
        """
        z:     [B,C,H,W] (or [B,2,Nr,Nt]) noisy sample
        sigma: [B,1,1,1]
        return: x0_hat same shape as z
        """
        _, _, c_in = self.get_scaling_for_boundary_condition(
            sigma=sigma, sigma_data=self.sigma_data, sigma_min=0.0
        )
        c_noise = 0.25 * torch.log(sigma[:, 0, 0, 0])  # [B]
        score_pred = self.net(z * c_in, c_noise)  # net outputs score in z-space
        x0_hat = z + (sigma ** 2) * score_pred # Tweedie's formula
        return x0_hat, score_pred

    @torch.no_grad()
    def x0_from_velocity(self, z, sigma):
        """
        z:     [B,C,H,W] noisy sample
        sigma: [B,1,1,1]
        return: x0_hat same shape as z
        """
        _, _, c_in = self.get_scaling_for_boundary_condition(
            sigma=sigma, sigma_data=self.sigma_data, sigma_min=0.0
        )
        c_noise = 0.25 * torch.log(sigma[:, 0, 0, 0])  # [B]
        v_pred = self.net(z * c_in, c_noise)  # net outputs v = (z-x0)/sigma
        x0_hat = z - sigma * v_pred
        return x0_hat, v_pred

    @torch.no_grad()
    def edm_sampling(self, H_noisy, sigma):
        """
        Use the model to predict the clean channel image based on noisy image
            H_noisy: [B,2,Nr,Nt] noisy channel image at noise level sigma
            sigma: [B, 1, 1, 1]
            return: the predicted image H0_hat, and the velocity:
        """
        c_skip, c_out, c_in = self.get_scaling_for_boundary_condition(sigma=sigma, sigma_data=self.sigma_data, sigma_min=0.0)
        c_noise = 0.25 * torch.log(sigma[:, 0, 0, 0]) # [B]
        H_pred = self.net(H_noisy * c_in, c_noise)
        H0_hat = c_skip * H_noisy + c_out * H_pred
        v = (H_noisy - H0_hat) / sigma # the velocity
        return H0_hat, v

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

    @torch.no_grad()
    def init_ema(self):
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            self.ema_params1[name] = p.detach().clone()
            self.ema_params2[name] = p.detach().clone()


