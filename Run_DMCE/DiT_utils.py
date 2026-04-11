
from math import pi

import torch
from torch import nn
import numpy as np

from einops import rearrange, repeat


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    # grid: [2, 1, grid_size_h, grid_size_w]
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    embed_dim: hidden size
    grid_size: int of the grid height and width, = sqrt(num_patches)
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first, [2, grid_size, grid_size], 2 array
    grid = np.stack(grid, axis=0) # [2, grid_size, grid_size]

    grid = grid.reshape([2, 1, grid_size, grid_size]) # [2, 1, grid_size, grid_size]
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_General(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    embed_dim: hidden size
    grid_size:
        -- int: for square image, = the grid height and width, = sqrt(num_patches)
        -- (grid_size_h, grid_size_w): for non-square image
    return:
    pos_embed: [grid_size_h*grid_size_w, embed_dim] or [1+grid_size_h*grid_size_w, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size_h = grid_size_w = grid_size
    else:
        assert len(grid_size) == 2
        grid_size_h, grid_size_w = grid_size

    # construct the grid coordinates:
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first, [2, grid_size_h, grid_size_w], 2 array
    grid = np.stack(grid, axis=0)  # [2, grid_size_h, grid_size_w]

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])  # [2, 1, grid_size_h, grid_size_w]

    # generate the 2D sin/cos positional encoding:
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    # deal with CLS tikens:
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)

    return pos_embed


class VisionRotaryEmbeddingFast_General(nn.Module):
    '''
    This is a general version of VisionRotaryEmbeddingFast,
    For square image, keep the same call method:
    rope = VisionRotaryEmbeddingFast(dim=64, pt_seq_len=16, ft_seq_len=16)
    For non-square image:
    rope = VisionRotaryEmbeddingFast(dim=64,
                                 pt_seq_len=(10, 14),
                                 ft_seq_len=(21, 28))
    '''
    def __init__(self,
                 dim,
                 pt_seq_len=16,  # the seq_len when pre-training, can be int = \sqrt(num_path), or (pt_h, pt_w) for non-square img.
                 ft_seq_len=None,  # the seq_len when fine-tuning, can be different from pt_seq_len
                 custom_freqs=None,
                 freqs_for='lang',
                 theta=10000,
                 max_freq=10,
                 num_freqs=1,
                 num_cls_token=0
                 ):
        super().__init__()

        # 1. freqs, same with the VisionRotaryEmbeddingFast
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            # the form in the standard RoPE, \freq_i = 10000^(-2i/d) = 1 / theta^(2i/d)
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim)) # shape: [dim/2]
        elif freqs_for == 'pixel': # for image
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        # 2. deal with the width and height of pt/ft, support int and (H, W)
        # for pre-training:
        if isinstance(pt_seq_len, int):
            pt_h = pt_w = pt_seq_len
        else:
            assert len(pt_seq_len) == 2
            pt_h, pt_w = pt_seq_len

        # for fine-tuning:
        if ft_seq_len is None:
            ft_h, ft_w = pt_h, pt_w
        elif isinstance(ft_seq_len, int):
            ft_h = ft_w = ft_seq_len
        else:
            assert len(ft_seq_len) == 2
            ft_h, ft_w = ft_seq_len

        # 3. construct the 1D angle matrix for both row and column
        # for row;
        # affine the range of ft_seq_len to the range of pt_seq_len
        t_h = torch.arange(ft_h) / ft_h * pt_h # [ft_h]
        freqs_h = torch.einsum('..., f -> ... f', t_h, freqs)  # [ft_h, dim/2]
        # each freq repeat 2 times, for the odd and even components,
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r=2)  # [ft_h, dim]

        # for column:
        t_w = torch.arange(ft_w) / ft_w * pt_w # [ft_w]
        freqs_w = torch.einsum('..., f -> ... f', t_w, freqs)  # [ft_w, dim/2]
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r=2)  # [ft_w, dim]

        # 4. Combine to 2D RoPE frequency table: [ft_h, ft_w, 2*dim]
        freqs_2d = broadcat(
            (freqs_h[:, None, :],  # [ft_h, 1, dim]
             freqs_w[None, :, :]),  # [1, ft_w, dim]
            dim=-1
        )  # -> [ft_h, ft_w, 2*dim]

        # 5. deal with the CLS and then flatten:
        if num_cls_token > 0:
            freqs_flat = freqs_2d.view(-1, freqs_2d.shape[-1])  # [N_patch, 2*dim]
            cos_img = freqs_flat.cos()
            sin_img = freqs_flat.sin()

            # prepend in-context cls token
            N_img, D = cos_img.shape
            cos_pad = torch.ones(num_cls_token, D, dtype=cos_img.dtype, device=cos_img.device)
            sin_pad = torch.zeros(num_cls_token, D, dtype=sin_img.dtype, device=sin_img.device)

            self.freqs_cos = torch.cat([cos_pad, cos_img], dim=0).cuda() # [N_cls+N_img, D]
            self.freqs_sin = torch.cat([sin_pad, sin_img], dim=0).cuda()
            # self.freqs_cos = torch.cat([cos_pad, cos_img], dim=0)  # [N_cls+N_img, D]
            # self.freqs_sin = torch.cat([sin_pad, sin_img], dim=0)
        else:
            # without CLS token, first cos/sin, then flatten to [N_img, D]
            self.freqs_cos = freqs_2d.cos().view(-1, freqs_2d.shape[-1]).cuda() # [N_img, D=2*dim]
            self.freqs_sin = freqs_2d.sin().view(-1, freqs_2d.shape[-1]).cuda()
            # self.freqs_cos = freqs_2d.cos().view(-1, freqs_2d.shape[-1]) # [N_img, D=2*dim]
            # self.freqs_sin = freqs_2d.sin().view(-1, freqs_2d.shape[-1])

    def forward(self, t):
        # x_rot = x * cos(theta) + R(x) * sin(theta)
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16, # the square of seq_len when pre-training, --- \sqrt(num_pathches)
        ft_seq_len=None, # the square of seq_len when fine-tuning, can be different from pt_seq_len
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        num_cls_token = 0
    ):
        super().__init__()
        # 1. construct the basic frequencies freqs, like the frequencies table in RoPE
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            # the form in the standard RoPE, \freq_i = 10000^(-2i/d) = 1 / theta^(2i/d)
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim)) # shape: [dim/2]
        elif freqs_for == 'pixel': # for image
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        # 2. for fine tuning deq_len
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len

        # the position of patches, affine the range of ft_seq_len to the range of pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len # shape: [ft_seq_len]

        # 3. construct the angle matrix
        #
        freqs = torch.einsum('..., f -> ... f', t, freqs) # [ft_seq_len, dim/2]
        # each freq repeat 2 times, for the odd and even components,
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2) # [ft_seq_len, dim]

        #4.  expand to 2D grid, and concat as 2D RoPE freqs:
        # freqs[:, None, :]: shape: [ft_seq_len, 1, dim] for row
        # freqs[None, :, :]: shape: [1, ft_seq_len, dim] for column
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1) # shape: [ft_seq_len, ft_seq_len, 2 * dim]

        # 5. if has CLS token, insert
        if num_cls_token > 0:
            freqs_flat = freqs.view(-1, freqs.shape[-1])  # [N_img, D]
            cos_img = freqs_flat.cos()
            sin_img = freqs_flat.sin()

            # prepend in-context cls token
            N_img, D = cos_img.shape
            cos_pad = torch.ones(num_cls_token, D, dtype=cos_img.dtype, device=cos_img.device)
            sin_pad = torch.zeros(num_cls_token, D, dtype=sin_img.dtype, device=sin_img.device)

            self.freqs_cos = torch.cat([cos_pad, cos_img], dim=0).cuda()  # [N_cls+N_img, D]
            self.freqs_sin = torch.cat([sin_pad, sin_img], dim=0).cuda()
        else:
            # without CLS token, first cos/sin, then flatten to [N_img, dim]
            self.freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).cuda()
            self.freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).cuda()

    def forward(self, t):
        # x_rot = x * cos(theta) + R(x) * sin(theta)
        return  t * self.freqs_cos + rotate_half(t) * self.freqs_sin




def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2) # [..., num_patch, dim] -> [..., num_patch, dim/2, 2]
    x1, x2 = x.unbind(dim = -1) # X1 AND X2: [..., num_patch, dim/2]
    x = torch.stack((-x2, x1), dim = -1) # [..., num_patch, dim/2, 2]
    # [..., num_patch, dim]
    return rearrange(x, '... d r -> ... (d r)')



def add_weight_decay(model, weight_decay=0, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or 'diffloss' in name:
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]