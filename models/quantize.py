from math import log2, ceil
from collections import namedtuple

import torch
from torch import nn, Tensor, einsum
import torch.nn.functional as F
from torch.nn import Module

from einops import rearrange, reduce, pack, unpack

# constants

Return = namedtuple('Return', ['quantized', 'indices', 'entropy_aux_loss'])

LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'batch_entropy', 'commitment'])

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# distance

def euclidean_distance_squared(x, y):
    x2 = reduce(x ** 2, '... n d -> ... n', 'sum')
    y2 = reduce(y ** 2, 'n d -> n', 'sum')
    xy = einsum('... i d, j d -> ... i j', x, y) * -2
    return rearrange(x2, '... i -> ... i 1') + y2 + xy

# entropy

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def entropy(prob):
    return -prob * log(prob)

# class

class LFQ(Module):
    def __init__(
        self,
        *,
        dim = None,
        codebook_size = None,
        entropy_loss_weight = 0.1,
        commitment_loss_weight = 1.,
        diversity_gamma = 2.5,
        straight_through_activation = nn.Identity(),
        num_codebooks = 1,
        keep_num_codebooks_dim = None,
        codebook_scale = 1.
    ):
        super().__init__()


        assert exists(dim) or exists(codebook_size), 'either dim or codebook_size must be specified for LFQ'
        assert not exists(codebook_size) or log2(codebook_size).is_integer(), f'your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})'

        codebook_size = default(codebook_size, lambda: 2 ** dim)
        codebook_dim = int(log2(codebook_size))

        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)

        self.project_in = nn.Linear(dim, codebook_dims) if dim != codebook_dims else nn.Identity()
        self.project_out = nn.Linear(codebook_dims, dim) if dim != codebook_dims else nn.Identity()

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim


        self.activation = straight_through_activation


        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight


        self.codebook_scale = codebook_scale

        self.commitment_loss_weight = commitment_loss_weight


        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.), persistent = False)


        all_codes = torch.arange(codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = self.bits_to_codes(bits)

        self.register_buffer('codebook', codebook, persistent = False)

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    @property
    def dtype(self):
        return self.codebook.dtype

    def indices_to_codes(
        self,
        indices,
        project_out = True
    ):
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... -> ... 1')


        bits = ((indices[..., None].int() & self.mask) != 0).to(self.dtype)

        codes = self.bits_to_codes(bits)

        codes = rearrange(codes, '... c d -> ... (c d)')



        if project_out:
            codes = self.project_out(codes)


        if is_img_or_video:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def forward(
        self,
        x,
        inv_temperature = 1.,
        return_loss_breakdown = False
    ):

        is_img_or_video = x.ndim >= 4


        if is_img_or_video:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack_one(x, 'b * d')

        assert x.shape[-1] == self.dim, f'expected dimension of {self.dim} but received {x.shape[-1]}'

        x = self.project_in(x)



        x = rearrange(x, 'b n (c d) -> b n c d', c = self.num_codebooks)


        original_input = x

        codebook_value = torch.ones_like(x) * self.codebook_scale
        print(codebook_value.shape)
        quantized = torch.where(x > 0, codebook_value, -codebook_value)

        if self.training:
            x = self.activation(x)
            x = x - x.detach() + quantized
        else:
            x = quantized


        indices = reduce((x > 0).int() * self.mask.int(), 'b n c d -> b n c', 'sum')


        if self.training:
            distance = euclidean_distance_squared(original_input, self.codebook)

            prob = (-distance * inv_temperature).softmax(dim = -1)

            per_sample_entropy = entropy(prob).mean()

            avg_prob = reduce(prob, 'b n c d -> b c d', 'mean')
            codebook_entropy = entropy(avg_prob).mean()

            entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
        else:

            entropy_aux_loss = per_sample_entropy = codebook_entropy = self.zero


        if self.training:
            commit_loss = F.mse_loss(original_input, quantized.detach())
        else:
            commit_loss = self.zero


        x = rearrange(x, 'b n c d -> b n (c d)')


        x = self.project_out(x)


        if is_img_or_video:
            x = unpack_one(x, ps, 'b * d')
            x = rearrange(x, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')


        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')


        aux_loss = entropy_aux_loss * self.entropy_loss_weight + commit_loss * self.commitment_loss_weight

        ret = Return(x, indices, aux_loss)

        if not return_loss_breakdown:
            return ret

        return ret, LossBreakdown(per_sample_entropy, codebook_entropy, commit_loss)
