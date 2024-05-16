from functools import wraps
import math, copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import pdb


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

    def __init__(self, size, self_attn, seq_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.seq_attn = seq_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, seq,src_mask=None, tgt_mask=None):
        "Follow Figure 1 (right) for connections."
        s = seq
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, attn_mask=tgt_mask)[0])
        x = self.sublayer[1](x, lambda x: self.seq_attn(x, s, s)[0])
        # attn = self.self_attn.attn

        # x3=torch.add(x1,x2)
        return self.sublayer[2](x, self.feed_forward)


class TransformerDec(nn.Module):
    def __init__(self):
        super().__init__()

        self.command_fcn = nn.Linear(512, 5)
        self.args_fcn = nn.Linear(512, 2 * 128)
        c = copy.deepcopy
        attn = nn.MultiheadAttention(num_heads=8, embed_dim=512, dropout=0.1, batch_first=True)
        #attn1 = nn.MultiheadAttention(num_heads=8, embed_dim=512, dropout=0.1, batch_first=True)

        ff = PositionwiseFeedForward(d_model=512, d_ff=1024, dropout=0.1)

        self.decoder_layers = clones(DecoderLayer(512, c(attn), c(attn), c(ff), dropout=0.1), 5)
        self.decoder_norm = nn.LayerNorm(512)
        # self.decoder_norm_parallel = nn.LayerNorm(512)
        # self.cls_embedding = nn.Embedding(52, 512)
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, 512))

    def forward(self, x,ref, src_mask=None, tgt_mask=None):
        for layer in self.decoder_layers:
            x = layer(x, ref, src_mask, tgt_mask)
        out = self.decoder_norm(x)
        N, S, _ = out.shape

        #############################################
        cmd_logits = self.command_fcn(out)
        args_logits = self.args_fcn(out)  # shape: bs, max_len, 8, 256
        args_logits = args_logits.reshape(N, S, 2, 128)

        return cmd_logits, args_logits



class TransformerRef(nn.Module):
    def __init__(self):
        super().__init__()

        self.fcn = nn.Linear(512, 504+2,bias=False)

        c = copy.deepcopy
        attn = nn.MultiheadAttention(num_heads=8, embed_dim=512, dropout=0.1, batch_first=True)

        attn1 = nn.MultiheadAttention(num_heads=8, embed_dim=512, dropout=0.1, batch_first=True)

        ff = PositionwiseFeedForward(d_model=512, d_ff=1024, dropout=0.1)

        self.decoder_layers = clones(DecoderLayer(512, c(attn), c(attn1), c(ff), dropout=0.1), 6)

        self.decoder_norm = nn.LayerNorm(512)

        self.ref_embed = nn.Embedding(504+2, 512)  # padding_idx=0 表示索引为0的位置将被视为填充索引


    def forward(self, x, ref, src_mask=None, tgt_mask=None):

        x=self.ref_embed(x+2)

        for layer in self.decoder_layers:
            x = layer(x, ref, src_mask, tgt_mask)
        out = self.decoder_norm(x)
        N, S, _ = out.shape

        #############################################

        ref_logits = self.fcn(out)  # shape: bs, max_len, 8, 256

        return ref_logits


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def generate_square_subsequent_mask():
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    sz=768
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    print(mask.shape)
    return mask




if __name__ == '__main__':
    from torch.autograd import Variable

    batch_size=3
    num_heads=8
    seq_len=784
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).unsqueeze(0).expand(batch_size * num_heads, -1, -1)
    #mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #print(mask)

    model = TransformerDec()
    args = torch.ones(3, 784, 512)

    args1 = torch.ones(3, 784, 512)



    #model(x=args, seq=args1, img=args1, tgt_mask=mask)

    from torchsummary import summary

    summary(model, [(784, 512),(784, 512),(784, 512)])


