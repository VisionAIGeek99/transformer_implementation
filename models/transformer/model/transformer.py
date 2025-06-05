import torch
from torch import nn

from models.transformer.layers.multi_head_attention import MultiHeadAttention

class Transformer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, 8)

    def forward(self, x):
        self.mha(x)