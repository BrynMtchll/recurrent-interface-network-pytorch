import math
from pathlib import Path
import random
from functools import partial
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
from torch.special import expm1
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import blobfile as bf


from torch.optim import Adam
from torchvision import transforms as T, utils

from beartype import beartype

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from rin_pytorch.attend import Attend

import numpy

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator, DistributedDataParallelKwargs

# helpers functions

def exists(x):
    return x is not None

def identity(x):
    return x

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def safe_div(numer, denom, eps = 1e-10):
    return numer / denom.clamp(min = eps)

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    num_sqrt = math.sqrt(num)
    return int(num_sqrt) == num_sqrt

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

# use layernorm without bias, more stable

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class MultiHeadedRMSNorm(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# positional embeds

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        norm = False,
        qk_norm = False,
        time_cond_dim = None
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.time_cond = None

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        self.norm = LayerNorm(dim) if norm else nn.Identity()

        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = MultiHeadedRMSNorm(dim_head, heads)
            self.k_norm = MultiHeadedRMSNorm(dim_head, heads)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(
        self,
        x,
        time = None
    ):
        h = self.heads
        x = self.norm(x)

        if exists(self.time_cond):
            assert exists(time)
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = torch.einsum('b h n d, b h n e -> b h d e', k, v)

        out = torch.einsum('b h d e, b h n d -> b h n e', context, q)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context = None,
        heads = 4,
        dim_head = 32,
        norm = False,
        norm_context = False,
        time_cond_dim = None,
        flash = False,
        qk_norm = False
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        self.time_cond = None

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.norm = LayerNorm(dim) if norm else nn.Identity()
        self.norm_context = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, hidden_dim * 2, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = MultiHeadedRMSNorm(dim_head, heads)
            self.k_norm = MultiHeadedRMSNorm(dim_head, heads)

        self.attend = Attend(flash = flash)

    def forward(
        self,
        x,
        context = None,
        time = None
    ):
        h = self.heads

        if exists(context):
            context = self.norm_context(context)

        x = self.norm(x)

        context = default(context, x)

        if exists(self.time_cond):
            assert exists(time)
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PEG(nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.ds_conv = nn.Conv2d(dim, dim, 3, padding = 1, groups = dim)

    def forward(self, x, hw):
        b, n, d = x.shape
        x = rearrange(x, 'b (h w) d -> b d h w', h = hw)
        x = self.ds_conv(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, time_cond_dim = None):
        super().__init__()
        self.norm = LayerNorm(dim)

        self.time_cond = None

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x, time = None):
        x = self.norm(x)

        if exists(self.time_cond):
            assert exists(time)
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        return self.net(x)

# model

class RINBlock(nn.Module):
    def __init__(
        self,
        dim,
        latent_self_attn_depth,
        dim_latent = None,
        final_norm = True,
        patches_self_attn = True,
        **attn_kwargs
    ):
        super().__init__()
        dim_latent = default(dim_latent, dim)

        self.latents_attend_to_patches = Attention(dim_latent, dim_context = dim, norm = True, norm_context = True, **attn_kwargs)
        self.latents_cross_attn_ff = FeedForward(dim_latent)

        self.latent_self_attns = nn.ModuleList([])
        for _ in range(latent_self_attn_depth):
            self.latent_self_attns.append(nn.ModuleList([
                Attention(dim_latent, norm = True, **attn_kwargs),
                FeedForward(dim_latent)
            ]))

        self.latent_final_norm = LayerNorm(dim_latent) if final_norm else nn.Identity()

        self.patches_peg = PEG(dim)
        self.patches_self_attn = patches_self_attn

        if patches_self_attn:
            self.patches_self_attn = LinearAttention(dim, norm = True, **attn_kwargs)
            self.patches_self_attn_ff = FeedForward(dim)

        self.patches_attend_to_latents = Attention(dim, dim_context = dim_latent, norm = True, norm_context = True, **attn_kwargs)
        self.patches_cross_attn_ff = FeedForward(dim)

    def forward(self, patches, hw, latents, t):
        patches = self.patches_peg(patches, hw) + patches

        # latents extract or cluster information from the patches

        latents = self.latents_attend_to_patches(latents, patches, time = t) + latents

        latents = self.latents_cross_attn_ff(latents, time = t) + latents

        # latent self attention

        for attn, ff in self.latent_self_attns:
            latents = attn(latents, time = t) + latents
            latents = ff(latents, time = t) + latents

        if self.patches_self_attn:
            # additional patches self attention with linear attention

            patches = self.patches_self_attn(patches, time = t) + patches
            patches = self.patches_self_attn_ff(patches) + patches

        # patches attend to the latents

        patches = self.patches_attend_to_latents(patches, latents, time = t) + patches

        patches = self.patches_cross_attn_ff(patches, time = t) + patches

        latents = self.latent_final_norm(latents)
        return patches, latents

class RIN(nn.Module):
    def __init__(
        self,
        dim,
        patch_size = 16,
        channels = 3,
        depth = 6,                      # number of RIN blocks
        latent_self_attn_depth = 2,     # how many self attentions for the latent per each round of cross attending from pixel space to latents and back
        dim_latent = None,              # will default to image dim (dim)
        num_latents = 256,              # they still had to use a fair amount of latents for good results (256), in line with the Perceiver line of papers from Deepmind
        learned_sinusoidal_dim = 16,
        latent_token_time_cond = False, # whether to use 1 latent token as time conditioning, or do it the adaptive layernorm way (which is highly effective as shown by some other papers "Paella" - Dominic Rampas et al.)
        dual_patchnorm = True,
        patches_self_attn = True,       # the self attention in this repository is not strictly with the design proposed in the paper. offer way to remove it, in case it is the source of instability
        **attn_kwargs
    ):
        super().__init__()
        dim_latent = default(dim_latent, dim)

        self.channels = channels # times 2 due to self-conditioning
        self.patch_size = patch_size
        pixel_patch_dim = channels * (patch_size ** 2)

        # time conditioning

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        time_dim = dim * 4
        fourier_dim = learned_sinusoidal_dim + 1

        self.latent_token_time_cond = latent_token_time_cond
        time_output_dim = dim_latent if latent_token_time_cond else time_dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_output_dim)
        )

        # pixels to patch and back


        self.to_patches = Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(pixel_patch_dim * 2),
            # nn.AdaptiveAvgPool2d((pixel_patch_dim * 2, pixel_patch_dim * 2)),
            nn.Linear(pixel_patch_dim * 2, dim),
            nn.LayerNorm(dim) if dual_patchnorm else None,
        )

        # axial positional embeddings, parameterized by an MLP

        pos_emb_dim = dim // 2
        print(pos_emb_dim)
        self.axial_pos_emb_height_mlp = nn.Sequential(
            Rearrange('... -> ... 1'),
            nn.Linear(1, pos_emb_dim),
            nn.SiLU(),
            nn.Linear(pos_emb_dim, pos_emb_dim),
            nn.SiLU(),
            nn.Linear(pos_emb_dim, dim)
        )

        self.axial_pos_emb_width_mlp = nn.Sequential(
            Rearrange('... -> ... 1'),
            nn.Linear(1, pos_emb_dim),
            nn.SiLU(),
            nn.Linear(pos_emb_dim, pos_emb_dim),
            nn.SiLU(),
            nn.Linear(pos_emb_dim, dim)
        )

        # nn.Parameter(torch.randn(2, patch_height_width, dim) * 0.02)
        self.to_pixels = nn.Sequential(
            LayerNorm(dim),
            # nn.Linear(dim, dim),
            # LayerNorm(dim),
            # nn.AdaptiveAvgPool2d((pixel_patch_dim, pixel_patch_dim)),
            nn.Linear(dim, pixel_patch_dim),
            # LayerNorm(pixel_patch_dim),
            # nn.Linear(pixel_patch_dim, pixel_patch_dim),
        )

        self.latents = nn.Parameter(torch.randn(num_latents, dim_latent))
        nn.init.normal_(self.latents, std = 0.02)

        self.init_self_cond_latents = nn.Sequential(
            FeedForward(dim_latent),
            LayerNorm(dim_latent)
        )

        nn.init.zeros_(self.init_self_cond_latents[-1].gamma)

        # the main RIN body parameters  - another attention is all you need moment

        if not latent_token_time_cond:
            attn_kwargs = {**attn_kwargs, 'time_cond_dim': time_dim}

        self.blocks = nn.ModuleList([RINBlock(dim, dim_latent = dim_latent, latent_self_attn_depth = latent_self_attn_depth, patches_self_attn = patches_self_attn, **attn_kwargs) for _ in range(depth)])

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x,
        time,
        img_height,
        img_width,
        x_self_cond = None,
        latent_self_cond = None,
        return_latents = False
    ):
        batch = x.shape[0]

        x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))

        x = torch.cat((x_self_cond, x), dim = 1)

        # prepare time conditioning

        t = self.time_mlp(time)

        # prepare latents

        latents = repeat(self.latents, 'n d -> b n d', b = batch)

        # the warm starting of latents as in the paper

        if exists(latent_self_cond):
            latents = latents + self.init_self_cond_latents(latent_self_cond)

        # whether the time conditioning is to be treated as one latent token or for projecting into scale and shift for adaptive layernorm

        if self.latent_token_time_cond:
            t = rearrange(t, 'b d -> b 1 d')
            latents = torch.cat((latents, t), dim = -2)

        # to patches

        stride = self.patch_size - 3
        patches = F.unfold(x, kernel_size=(self.patch_size, self.patch_size), stride=(stride, stride))
        patches = rearrange(patches, 'b p l -> b l p')
        patches = self.to_patches(patches)

        # ((h - k - 1) / s + 1) * (((w - k - 1) / s) + 1)
        lh = round((x.shape[-2] - self.patch_size - 1) / stride) + 1
        lw = round((x.shape[-1] - self.patch_size - 1) / stride) + 1
        xh = x.shape[-2]
        xw = x.shape[-1]

        #(h - 1) * s = (ph - k - 1)
        #(h - 1) * s + 1 + k = h
        #(w - 1) * s + 1 + k = w


        height_range = torch.linspace(0., 1., steps = lh, device = self.device)
        width_range = torch.linspace(0., 1., steps = lw, device = self.device)

        pos_emb_h, pos_emb_w = self.axial_pos_emb_height_mlp(height_range), self.axial_pos_emb_width_mlp(width_range)
        pos_emb = rearrange(pos_emb_h, 'i d -> i 1 d') + rearrange(pos_emb_w, 'j d -> 1 j d')
        # print(lh)
        # print(lw)
        # print(x.shape)
        # print(patches.shape)
        # print(pos_emb.shape)
        # print(rearrange(pos_emb, 'i j d -> (i j) d').shape)

        patches = patches + rearrange(pos_emb, 'i j d -> (i j) d')

        # the recurrent interface network body

        for block in self.blocks:
            patches, latents = block(patches, lh, latents, t)

        # to pixels
        pixels = self.to_pixels(patches)
        pixels = rearrange(pixels, 'b l p -> b p l')
        pixels = F.fold(pixels, output_size=(xh, xw), kernel_size=(self.patch_size, self.patch_size), stride=(stride, stride))
        
        # pixels = rearrange(pixels, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)', p1 = self.patch_size, p2 = self.patch_size, h = lh, w = lw)
        if not return_latents:
            return pixels

        # remove time conditioning token, if that is the settings

        if self.latent_token_time_cond:
            latents = latents[:, :-1]

        return pixels, latents

# normalize and unnormalize image

def normalize_img(x):
    return x * 2 - 1

def unnormalize_img(x):
    return (x + 1) * 0.5

# normalize variance of noised image, if scale is not 1

def normalize_img_variance(x, eps = 1e-5):
    std = reduce(x, 'b c h w -> b 1 1 1', partial(torch.std, unbiased = False))
    return x / std.clamp(min = eps)

# helper functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# noise schedules

def simple_linear_schedule(t, clip_min = 1e-9):
    return (1 - t).clamp(min = clip_min)

def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = torch.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)

# converting gamma to alpha, sigma or logsnr

def gamma_to_alpha_sigma(gamma, scale = 1):
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)

def gamma_to_log_snr(gamma, scale = 1, eps = 1e-5):
    return log(gamma * (scale ** 2) / (1 - gamma), eps = eps)

# gaussian diffusion

@beartype
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: RIN,
        *,
        timesteps = 1000,
        use_ddim = True,
        noise_schedule = 'sigmoid',
        objective = 'v',
        schedule_kwargs: dict = dict(),
        time_difference = 0.,
        min_snr_loss_weight = True,
        min_snr_gamma = 5,
        train_prob_self_cond = 0.9,
        scale = 1.                      # this will be set to < 1. for better convergence when training on higher resolution images
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels

        assert objective in {'x0', 'eps', 'v'}, 'objective must be either predict x0 or noise'
        self.objective = objective

        if noise_schedule == "linear":
            self.gamma_schedule = simple_linear_schedule
        elif noise_schedule == "cosine":
            self.gamma_schedule = cosine_schedule
        elif noise_schedule == "sigmoid":
            self.gamma_schedule = sigmoid_schedule
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        # the main finding presented in Ting Chen's paper - that higher resolution images requires more noise for better training

        assert scale <= 1, 'scale must be less than or equal to 1'
        self.scale = scale
        self.maybe_normalize_img_variance = normalize_img_variance if scale < 1 else identity

        # gamma schedules

        self.gamma_schedule = partial(self.gamma_schedule, **schedule_kwargs)

        self.timesteps = timesteps
        self.use_ddim = use_ddim

        # proposed in the paper, summed to time_next
        # as a way to fix a deficiency in self-conditioning and lower FID when the number of sampling timesteps is < 400

        self.time_difference = time_difference

        # probability for self conditioning during training

        self.train_prob_self_cond = train_prob_self_cond

        # min snr loss weight

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

    @property
    def device(self):
        return next(self.model.parameters()).device

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    @torch.no_grad()
    def ddpm_sample(self, shape, time_difference = None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        img = torch.randn(shape, device=device)

        x_start = None
        last_latents = None
        print("sampling...")
        for time, time_next in time_pairs:

            # add the time delay

            time_next = (time_next - self.time_difference).clamp(min = 0.)

            noise_cond = time

            # get predicted x0

            maybe_normalized_img = self.maybe_normalize_img_variance(img)
            model_output, last_latents = self.model(maybe_normalized_img, noise_cond, shape[2], shape[3], x_start, last_latents, return_latents = True)

            # get log(snr)

            gamma = self.gamma_schedule(time)
            gamma_next = self.gamma_schedule(time_next)
            gamma, gamma_next = map(partial(right_pad_dims_to, img), (gamma, gamma_next))

            # get alpha sigma of time and next time

            alpha, sigma = gamma_to_alpha_sigma(gamma)
            alpha_next, sigma_next = gamma_to_alpha_sigma(gamma_next)

            # calculate x0 and noise

            if self.objective == 'x0':
                x_start = model_output

            elif self.objective == 'eps':
                x_start = safe_div(img - sigma * model_output, alpha)

            elif self.objective == 'v':
                x_start = alpha * img - sigma * model_output

            # clip x0

            x_start.clamp_(-1., 1.)

            # derive posterior mean and variance

            log_snr, log_snr_next = map(gamma_to_log_snr, (gamma, gamma_next))

            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # get noise

            noise = torch.where(
                rearrange(time_next > 0, 'b -> b 1 1 1'),
                torch.randn_like(img),
                torch.zeros_like(img)
            )

            img = mean + (0.5 * log_variance).exp() * noise

        return unnormalize_img(img)

    @torch.no_grad()
    def ddim_sample(self, shape, time_difference = None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        img = torch.randn(shape, device = device)

        x_start = None
        last_latents = None
        print("sampling...")
        for times, times_next in time_pairs:

            # get times and noise levels

            gamma = self.gamma_schedule(times)
            gamma_next = self.gamma_schedule(times_next)

            padded_gamma, padded_gamma_next = map(partial(right_pad_dims_to, img), (gamma, gamma_next))

            alpha, sigma = gamma_to_alpha_sigma(padded_gamma)
            alpha_next, sigma_next = gamma_to_alpha_sigma(padded_gamma_next)

            # add the time delay

            times_next = (times_next - time_difference).clamp(min = 0.)

            # predict x0

            maybe_normalized_img = self.maybe_normalize_img_variance(img)
            model_output, last_latents = self.model(maybe_normalized_img, times, shape[2], shape[3], x_start, last_latents, return_latents = True)

            # calculate x0 and noise

            if self.objective == 'x0':
                x_start = model_output

            elif self.objective == 'eps':
                x_start = safe_div(img - sigma * model_output, alpha)

            elif self.objective == 'v':
                x_start = alpha * img - sigma * model_output

            # clip x0

            x_start.clamp_(-1., 1.)

            # get predicted noise

            pred_noise = safe_div(img - alpha * x_start, sigma)

            # calculate x next

            img = x_start * alpha_next + pred_noise * sigma_next

        return unnormalize_img(img)

    @torch.no_grad()
    def sample(self, img_height = 128, img_width = 128, batch_size = 16):
        channels = self.channels
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample
        return sample_fn((batch_size, channels, img_height, img_width))

    def forward(self, img, *args, **kwargs):
        batch, c, h, w, device = *img.shape, img.device
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # sample random times

        times = torch.zeros((batch,), device = device).float().uniform_(0, 1.)

        # convert image to bit representation

        img = normalize_img(img)

        # noise sample

        noise = torch.randn_like(img)

        gamma = self.gamma_schedule(times)
        padded_gamma = right_pad_dims_to(img, gamma)
        alpha, sigma =  gamma_to_alpha_sigma(padded_gamma, self.scale)

        noised_img = alpha * img + sigma * noise

        noised_img = self.maybe_normalize_img_variance(noised_img)

        # in the paper, they had to use a really high probability of latent self conditioning, up to 90% of the time
        # slight drawback

        self_cond = self_latents = None

        if random.random() < self.train_prob_self_cond:
            with torch.no_grad():
                model_output, self_latents = self.model(noised_img, times, h, w, return_latents = True)
                self_latents = self_latents.detach()

                if self.objective == 'x0':
                    self_cond = model_output

                elif self.objective == 'eps':
                    self_cond = safe_div(noised_img - sigma * model_output, alpha)

                elif self.objective == 'v':
                    self_cond = alpha * noised_img - sigma * model_output

                self_cond.clamp_(-1., 1.)
                self_cond = self_cond.detach()

        # predict and take gradient step

        pred = self.model(noised_img, times, h, w, self_cond, self_latents)

        if self.objective == 'eps':
            target = noise

        elif self.objective == 'x0':
            target = img

        elif self.objective == 'v':
            target = alpha * noise - sigma * img

        loss = F.mse_loss(pred, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        # min snr loss weight

        snr = (alpha * alpha) / (sigma * sigma)
        maybe_clipped_snr = snr.clone()

        if self.min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = self.min_snr_gamma)

        if self.objective == 'eps':
            loss_weight = maybe_clipped_snr / snr

        elif self.objective == 'x0':
            loss_weight = maybe_clipped_snr

        elif self.objective == 'v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        return (loss * loss_weight).mean()

# dataset classes

class Dataset(Dataset):
    def __init__(
        self,
        img_path,
        resolution,
        patch_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        with bf.BlobFile(img_path, "rb") as f:
            self.img = Image.open(f)
            self.img = self.img.convert("RGB")
            self.img.load()

        maybe_convert_fn = partial(convert_image_to, convert_image_to) if exists(convert_image_to) else nn.Identity()
        self.resolution = resolution
        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(self.resolution),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(self.resolution),
            T.ToTensor()
        ])
        self.img = self.transform(self.img)
        self.patch_size = patch_size

        # self.rand_interps = []
        # print(self.img.shape)
        # for i in range(2000):
        #     new_resolution = self.resolution * random.uniform(0.5, 1.25)
        #     new_resolution = round(patch_size * (new_resolution // patch_size))
        #     data = F.interpolate(self.img, (new_resolution, new_resolution), mode="bicubic")
        #     self.rand_interps.append(data)
            

    # def random_resize(self, data, patch_size):
    #     new_resolution = self.resolution * random.uniform(0.5, 1.25)
    #     curr_h = round(data.shape[2] * new_resolution)
    #     curr_w = round(data.shape[3] * new_resolution)
    #     curr_h, curr_w = patch_size * (curr_h // patch_size), patch_size * (curr_w // patch_size)
    #     data = F.interpolate(data, (curr_h, curr_w), mode="bicubic")

    #     return data

    def __len__(self):
        return 10000

    def interp(self, data):
        new_resolution_w = self.resolution * random.uniform(0.5, 1.5)
        new_resolution_h = self.resolution * random.uniform(0.5, 1.5)

        new_resolution_w = round((self.patch_size - 3) * (new_resolution_w // (self.patch_size - 3))) + self.patch_size + 1
        new_resolution_h = round((self.patch_size - 3) * (new_resolution_h // (self.patch_size - 3))) + self.patch_size + 1

        data = F.interpolate(data, (new_resolution_h, new_resolution_w), mode="bicubic")
        return data
    
    def random_crop(self, data): 
        new_h = int(data.shape[2] * random.uniform(0.5, 1))
        new_w = int(data.shape[3] * random.uniform(0.5, 1))
        # new_h = round(self.patch_size * (new_h // self.patch_size))
        # new_w = round(self.patch_size * (new_w // self.patch_size))

        offset_h = int((data.shape[2] - new_h) * random.uniform(0, 1))
        offset_w = int((data.shape[3] - new_w) * random.uniform(0, 1))
        # offset_h = round(self.patch_size * (offset_h // self.patch_size))
        # offset_w = round(self.patch_size * (offset_w // self.patch_size))

        return data[:,:,offset_h: offset_h + new_h, offset_w: offset_w + new_w]

    def __getitem__(self, index):
        return self.img

# trainer class

@beartype
class Trainer(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion,
        folder,
        *,
        resolution = 128,
        train_batch_size = 16,
        patch_size = 8,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        max_grad_norm = 1.,
        ema_update_every = 10,
        ema_decay = 0.995,
        betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 16,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no',
            kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
        )

        self.model = diffusion_model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        self.patch_size = patch_size

        # dataset and dataloader

        self.ds = Dataset(folder, resolution = resolution, patch_size=patch_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = betas)

        # for logging results in a folder periodically

        self.results_folder = Path(results_folder)

        if self.accelerator.is_local_main_process:
            self.results_folder.mkdir(exist_ok = True)

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)


        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step + 1,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=torch.device('cpu'))

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        print("training...")

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    # data = self.ds.random_crop(data)
                    data = self.ds.interp(data)

                    with accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                # save milestone on every local main process, sample only on global main process

                if accelerator.is_local_main_process:
                    milestone = self.step // self.save_and_sample_every
                    save_and_sample = self.step != 0 and self.step % self.save_and_sample_every == 0
                    
                    if accelerator.is_main_process:
                        self.ema.to(device)
                        self.ema.update()

                        if save_and_sample:
                            self.ema.ema_model.eval()

                            scales_h = [129, 249, 514, 249, 514]
                            scales_w = [129, 249, 249, 514, 514]
                            for i in range(len(scales_h)):
                                with torch.no_grad():
                                    batches = num_to_groups(self.num_samples, self.batch_size)
                                    all_images_list = list(map(lambda n: self.ema.ema_model.sample(img_height=scales_h[i], img_width = scales_w[i], batch_size=n), batches))
                                all_images = torch.cat(all_images_list, dim = 0)
                                utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}-scale-{scales_h[i]}x{scales_w[i]}.png'), nrow = int(math.sqrt(self.num_samples)))

                    if save_and_sample:
                        self.save(milestone)

                self.step += 1
                pbar.update(1)

        accelerator.print('training complete')
