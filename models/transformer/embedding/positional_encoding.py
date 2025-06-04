import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        # input data shape
        # [batch_size, sequence_length]

        super().__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

        # positional encoding visualization for debug
        # pe = self.encoding[:].cpu().numpy() 
        # plt.figure(figsize=(14, 6))
        # sns.heatmap(pe, cmap='RdBu', cbar=True)
        # plt.xlabel("Embedding Dimension")
        # plt.ylabel("Position")
        # plt.title("Positional Encoding Heatmap")
        # plt.savefig("./positional_encoding_visualization.png")


    def forward(self, x):
        # self.encoding

        batch_size, seq_len = x.shape

        return self.encoding[:seq_len, :]