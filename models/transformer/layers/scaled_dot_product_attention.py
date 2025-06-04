import torch
from torch import nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, q, k, v, mask=None):

        k_t = k.transpose
        pass