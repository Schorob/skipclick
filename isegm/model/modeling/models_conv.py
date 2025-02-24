from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle


class MLP(nn.Module):
    """
    Take from the Conv2Former Repository:
    https://github.com/HVision-NKU/Conv2Former/blob/main/convmod.py
    """
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x


class ConvMod(nn.Module):
    """
    Take from the Conv2Former Repository:
    https://github.com/HVision-NKU/Conv2Former/blob/main/convmod.py
    """
    def __init__(self, dim):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)

        return x


class Block(nn.Module):
    """
    Take from the Conv2Former Repository:
    https://github.com/HVision-NKU/Conv2Former/blob/main/convmod.py
    """
    def __init__(self, dim, mlp_ratio=4., drop_path=0.):
        super().__init__()

        self.attn = ConvMod(dim)
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class LayerNorm(nn.Module):
    r"""
    Take from the Conv2Former Repository:
    https://github.com/HVision-NKU/Conv2Former/blob/main/convmod.py

    [Which, again, has been taken] from ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SimpleModulationStack(nn.Module):

    def __init__(self, depth=4, embed_dim=768, img_size=448, patch_size=(14, 14), **kwargs):
        super().__init__()
        self.blocks = nn.Sequential(*[
            Block(embed_dim, mlp_ratio=4) for _ in range(depth)
        ])
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size

    def forward(self, x, additional_features, grid_size):
        # x.shape == additional_features.shape == [B, grid_size[0]*grid_size[1], dim]
        x_shape = x.shape
        if additional_features is not None:
            x += additional_features

        x = torch.transpose(x, 2, 1) # [B, dim, grid_size[0]*grid_size[1]]
        x = x.reshape(x_shape[0], self.embed_dim, grid_size[0], grid_size[1])  # [B, dim, grid_size[0], grid_size[1]]

        x = self.blocks(x) # <-- The actual computation of the convolutional modulation blocks

        x = x.reshape(x_shape[0], self.embed_dim, grid_size[0] * grid_size[1])
        x = torch.transpose(x, 2, 1)
        return x

