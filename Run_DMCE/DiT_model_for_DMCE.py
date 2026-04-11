
# --------------------------------------------------------
# References:
# JiT: https://github.com/LTH14/JiT?tab=readme-ov-file
# --------------------------------------------------------
'''
Note: this file is similar to our file: DiT_Models/DiT_model.py,
except add a "self.device = device" in the DiT_NN init function, as create the diffusion model in DMCE files require this variable
'''

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Union

from DiT_utils import RMSNorm, get_2d_sincos_pos_embed_General, VisionRotaryEmbeddingFast_General

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: tuple, patch_size: tuple, in_channels: int = 2, embed_dim: int = 1024):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape  # [1, 3, N_t, N_r]
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x) # x: [B, C, H, W] -> [B, embed_dim, H/patch_size, W/patch_size]
        x = x.reshape(x.shape[0], x.shape[1], -1) # -> [B, embed_dim, num_patches]
        x = x.permute(0, 2, 1)  # [B, C, num_patches] -> [B, num_patches, C]

        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp =  nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        # self.mlp = nn.Sequential(
        #     nn.Linear(frequency_embedding_size, hidden_size, bias=True),
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, hidden_size, bias=True),
        # )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None] # [N, 1] * [1, half] ==> [N, half]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)  # query: [B, num_heads, N, C // num_heads]
    scale_factor = 1 / math.sqrt(query.size(-1)) # \sqrt{d}
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()
    # attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype)

    # with torch.cuda.amp.autocast(enabled=False):
    with torch.amp.autocast(device_type='cuda', enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    # attn_weight = F.dropout(attn_weight, p=dropout_p, training=self.training)
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape # [B, num_patches, hidden_size]
        # [B, N, C] -> [B, N, 3C] -> [B, N, 3, num_heads, C // num_heads] -> [3, B, num_heads, N, C // num_heads]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q) # [B, num_heads, N, C // num_heads]
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.) # [B, num_heads, N, C // num_heads]

        x = x.transpose(1, 2).reshape(B, N, C) #

        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of DiT, follows the JiT
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size[0] * patch_size[1] * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    # @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    # @torch.compile
    def forward(self, x,  c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiT_NN(nn.Module):
    def __init__(self,
                 img_size: tuple = (64, 16),
                 in_channels: int = 2,
                 patch_size: tuple = (4, 4),
                 hidden_size: int = 1024,
                 num_heads: int = 8,
                 depth: int = 4,
                 mlp_ratio: int = 4,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 device: Union[str, torch.device] = 'cuda',
                 ):
        super().__init__()
        self.device = device
        self.img_size = img_size # [N_r, N_t]
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_head = num_heads

        # time embed:
        self.t_embedder = TimestepEmbedder(hidden_size)

        # image embed: img_size: tuple, patch_size: tuple, in_channels: int = 2, embed_dim: int = 1024
        self.x_embedder = PatchEmbedding(img_size, patch_size, in_channels, hidden_size)

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False) # [B, num_patches, hidden_size]

        # Rope:
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # hw_seq_len = input_size // patch_size #
        self.feat_rope = VisionRotaryEmbeddingFast_General(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )

        # transformer
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        # linear predict
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed: # [1, num_patches, hidden_size],
        # TODO: define the gride size for both W and H, not identical
        pos_embed = get_2d_sincos_pos_embed_General(self.pos_embed.shape[-1],
                                                    (int(self.img_size[0]/self.patch_size[0]),
                                                     int(self.img_size[1]/self.patch_size[1]))
                                                    )
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        # w2 = self.x_embedder.proj2.weight.data
        # nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp.weight, std=0.02)
        # nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: [B, num_patches, num_patch_w*patch_size_w*C]
        imgs: (B, C, H, W)
        """
        num_out_channels = self.out_channels
        patch_size_h = self.patch_size[0]
        patch_size_w = self.patch_size[1]
        # TODO: revise -----
        num_patch_h = self.img_size[0] // patch_size_h
        num_patch_w = self.img_size[1] // patch_size_w
        # h = w = int(x.shape[1] ** 0.5)
        assert num_patch_h * num_patch_w == x.shape[1]

        # [B, num_patches, num_patch_w*patch_size_w*C] -> [B, num_patch_h, num_patch_w, patch_size_h, patch_size_w, c]
        x = x.reshape(shape=(x.shape[0], num_patch_h, num_patch_w, patch_size_h, patch_size_w, num_out_channels))
        # [B, num_patch_h, num_patch_w, patch_size_h, patch_size_w, c] -> [B, c, num_patch_h, patch_size_h, num_patch_w, patch_size_w]
        x = torch.einsum('nhwpqc->nchpwq', x)
        # [B, c, num_patch_h, patch_size_h, num_patch_w, patch_size_w] -> [B, c, H, W]
        imgs = x.reshape(shape=(x.shape[0], num_out_channels, num_patch_h * patch_size_h, num_patch_w * patch_size_w))
        return imgs

    def forward(self, x, t):
        """
        x: [B, C, N_r, N_t]
        t: (B,): time step
        """
        # class and time embeddings
        t_emb = self.t_embedder(t) # [B, hidden_size]

        # forward DiT
        x = self.x_embedder(x) # [B, num_patch, hidden_size]
        x += self.pos_embed

        for i, block in enumerate(self.blocks):
            x = block(x, t_emb, self.feat_rope)

        x = self.final_layer(x, t_emb) # [B, num_patches, hidden_size] -> [B, num_patches, patch_size[0] * patch_size[1] * out_channels]
        output = self.unpatchify(x)

        return output

