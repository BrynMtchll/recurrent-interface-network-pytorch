from rin_pytorch.rin_pytorch import GaussianDiffusion, RIN, Trainer

import os

patch_size = 8

model = RIN(
    dim = 512,                  # model dimensions
    patch_size = patch_size,             # patch size
    depth = 6,                  # depth
    num_latents = 256,          # number of latents. they used 256 in the paper
    dim_latent = 1024,           # can be greater than the image dimension (dim) for greater capacity
    latent_self_attn_depth = 4, # number of latent self attention blocks per recurrent step, K in the paper
).cuda()

diffusion = GaussianDiffusion(
    model,
    timesteps = 400,
    noise_schedule = 'sigmoid',
    train_prob_self_cond = 0.9,  # how often to self condition on latents
    scale = 0.9                   # this will be set to < 1. for more noising and leads to better convergence when training on higher resolution images (512, 1024) - input noised images will be auto variance normalized
).cuda()
trainer = Trainer(
    diffusion,
    '/content/recurrent-interface-network-pytorch/data/pyramids.png',
    num_samples = 16,
    resolution = 256,
    train_batch_size = 8,
    patch_size = patch_size,
    gradient_accumulate_every = 4,
    train_lr = 1e-3,
    save_and_sample_every = 2000,
    train_num_steps = 100000,         # total training steps
    ema_decay = 0.995,                # exponential moving average decay
)

trainer.train()