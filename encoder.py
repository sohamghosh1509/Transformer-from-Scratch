import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import generate_positional_encoding, MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_heads=8, dropout=0.1):
        
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
                          nn.Linear(embed_dim, hidden_dim),
                          nn.ReLU(),
                          nn.Linear(hidden_dim, embed_dim)
        )
              
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        
        attention_out = self.attention(x,x,x, mask=mask)
        x = self.norm1(x + self.dropout(attention_out))

        feed_fwd_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(feed_fwd_out))

        return x


class TransformerEncoder(nn.Module):
    
    def __init__(self, src_vocab_size, embed_dim, max_seq_len, hidden_dim, num_layers=2, n_heads=8, dropout=0.2):
        super(TransformerEncoder, self).__init__()
        
        self.embedding_layer = nn.Embedding(src_vocab_size, embed_dim)
        self.positional_encoding = generate_positional_encoding(embed_dim, max_seq_len)

        self.layers = nn.ModuleList([EncoderBlock(embed_dim, hidden_dim, n_heads) for _ in range(num_layers)])
    
    def forward(self, x, mask):
        batch_size, seq_len = x.size()
        device = x.device
        self.positional_encoding = self.positional_encoding.to(device)
        embed_out = self.embedding_layer(x)
        out = embed_out + self.positional_encoding[:, :seq_len, :]
        for layer in self.layers:
            out = layer(out, mask)

        return out  #32x10x512
