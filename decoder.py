import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import generate_positional_encoding, MultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_heads=8, dropout=0.1):
        super(DecoderBlock, self).__init__()
        
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
                          nn.Linear(embed_dim, hidden_dim),
                          nn.ReLU(),
                          nn.Linear(hidden_dim, embed_dim)
        )
              
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, mask1, mask2):
        
        attention_out = self.attention(x ,x, x, mask=mask1) #32x10x512
        x = self.norm1(x + self.dropout(attention_out))
        cross_attn_output = self.attention(enc_out, x, enc_out, mask=mask2)
        x = self.norm2(x + self.dropout(cross_attn_output))
        feed_fwd_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(feed_fwd_out))
        return x
        
class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, max_seq_len, hidden_dim, num_layers=6, n_heads=8, dropout=0.2):
        super(TransformerDecoder, self).__init__()
        
        self.embedding_layer = nn.Embedding(target_vocab_size, embed_dim)
        self.positional_encoding = generate_positional_encoding(embed_dim,max_seq_len)

        self.layers = nn.ModuleList([DecoderBlock(embed_dim, hidden_dim, n_heads=8) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
            
    def forward(self, x, enc_out, mask1, mask2):            
        batch_size, seq_len = x.size()
        device = x.device
#         pad_mask = generate_padding_mask(x).to(device)
        self.positional_encoding = self.positional_encoding.to(device)
        embed_out = self.embedding_layer(x)
        out = embed_out + self.positional_encoding[:, :seq_len, :]
     
        for layer in self.layers:
            out = layer(out, enc_out, mask1, mask2) 

        out = self.fc_out(out)

        return out