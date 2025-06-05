from torch import nn 


from models.transformer.embedding.token_embedding import TokenEmbedding
from models.transformer.embedding.positional_encoding import PositionalEncoding

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, drop_prob: float, device):
        """
        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model input 
        """
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)
        self.device = device

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)

        return self.drop_out(tok_emb + pos_emb)