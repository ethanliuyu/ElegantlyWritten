
import torch
from torch import nn, einsum
import math, copy
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: [x_len, batch_size, emb_size]
        :return: [x_len, batch_size, emb_size]
        """
        x = x + self.pe[:x.size(0), :].to(x.device)
        return self.dropout(x)

class SVGEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        n_args=2

        self.command_embed = nn.Embedding(5, 512,padding_idx=0) #padding_idx=0 表示索引为0的位置将被视为填充索引
        self.arg_embed = nn.Embedding(128, 256,padding_idx=0)  #padding_idx=0 表示索引为0的位置将被视为填充索引
        self.embed_fcn = nn.Linear(256 * n_args, 512)

        d_model=512
        max_seq_len=180

        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_seq_len)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.command_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.arg_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.embed_fcn.weight, mode="fan_in")


    def forward(self, commands, args):

        S, GN = commands.shape

        src = self.command_embed(commands.long()+3).squeeze() + \
            self.embed_fcn(self.arg_embed((args+3).long()).view(S, GN, -1)) # shift due to -3 PAD_VAL

        src = self.pos_encoding(src)

        return src




def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(F.relu(self.dropout(self.w_1(x))))


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        x_norm = self.norm(x)
        return x + self.dropout(sublayer(x_norm))  # + self.augs(x_norm)





class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x,padding_mask=None):
        "Follow Figure 1 (right) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, key_padding_mask=padding_mask)[0])
        #attn = self.self_attn.attn
        return self.sublayer[1](x, self.feed_forward)


class TransformerEnc(nn.Module):
    def __init__(self):
        super().__init__()

        c = copy.deepcopy

        attn = nn.MultiheadAttention(num_heads=8, embed_dim=512, dropout=0.1,batch_first=True)

        ff = PositionwiseFeedForward(d_model=512, d_ff=1024, dropout=0.1)
        self.decoder_layers = clones(DecoderLayer(512, c(attn),c(ff), dropout=0.1), 6)
        self.decoder_norm = nn.LayerNorm(512)

    def forward(self, x,padding_mask=None):

        for layer in self.decoder_layers:
            x= layer(x,padding_mask)
        out = self.decoder_norm(x)

        return out
