import torch.nn.functional as F
import blobfile as bf
from PIL import Image
from torchvision import transforms as T, utils
import lpips
from SIFID.sifid_score import calculate_activation_statistics, calculate_frechet_distance
from SIFID.inception import InceptionV3
from torchvision import transforms as T, utils
import torch
from rin_pytorch.rin_pytorch import GaussianDiffusion, RIN, Trainer

model = RIN(
    dim = 512,                  # model dimensions
    patch_size = 8,             # patch size
    depth = 6,                  # depth
    num_latents = 256,          # number of latents. they used 256 in the paper
    dim_latent = 1024,           # can be greater than the image dimension (dim) for greater capacity
    latent_self_attn_depth = 4, # number of latent self attention blocks per recurrent step, K in the paper
).cuda()

diffusion = GaussianDiffusion(
    model,
    timesteps = 400,
    noise_schedule = 'cosine',
    train_prob_self_cond = 0.9,  # how often to self condition on latents
    scale = 1.                   # this will be set to < 1. for more noising and leads to better convergence when training on higher resolution images (512, 1024) - input noised images will be auto variance normalized
).cuda()



trainer = Trainer(
    diffusion,
    '/content/recurrent-interface-network-pytorch/data/comp4528-mini-proj-image.png',
    num_samples = 16,
    resolution = 256,
    train_batch_size = 8,
    patch_size = 8,
    gradient_accumulate_every = 2,
    train_lr = 1e-3,
    save_and_sample_every = 2000,
    train_num_steps = 100000,         # total training steps
    ema_decay = 0.995,                # exponential moving average decay
)

# model num
trainer.load(15)

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[64]

model = InceptionV3([block_idx]).cuda()

cuda = False

resolutions = [249, 514, 1029]
batch_size = 8


for resolution in resolutions:
  sampled_imgs = diffusion.sample(img_height = resolution, img_width = resolution, batch_size = batch_size)
  # images_cat = torch.cat(sampled_images, dim = 0)

  with bf.BlobFile('/content/recurrent-interface-network-pytorch/data/comp4528-mini-proj-image.png', "rb") as f:
      target_img = Image.open(f)
      target_img = target_img.convert("RGB")
      target_img.load()

  transform = T.Compose([
      T.Resize(resolution),
      T.CenterCrop(resolution),
      T.ToTensor()
  ])
  target_img = transform(target_img)
  target_img = target_img.to('gpu')

  utils.save_image(target_img, f'/content/recurrent-interface-network-pytorch/eval/target-{resolution}.png')

  mse_total = 0
  lpips_loss_total = 0
  sifid_values_total = 0

  for i in range(batch_size):
    mse = F.mse_loss(sampled_imgs[i], target_img, reduction = 'none')
  # mse = reduce(mse, 'b ... -> b', 'mean')
    mse_total += mse.mean().sqrt()

    loss_fn = lpips.LPIPS(net='alex')
    lpips_loss = loss_fn(sampled_imgs[i], target_img)
    lpips_loss_total += lpips_loss.mean()

    # print(sampled_img.shape)
    utils.save_image(sampled_imgs[i], f'/content/recurrent-interface-network-pytorch/eval/sample-{i}-{resolution}.png')

    m1, s1 = calculate_activation_statistics([f'/content/recurrent-interface-network-pytorch/eval/sample-{i}-{resolution}.png'], model, batch_size, 64, cuda)
    m2, s2 = calculate_activation_statistics([f'/content/recurrent-interface-network-pytorch/eval/target-{resolution}.png'], model, batch_size, 64, cuda)
    sifid_values_total += calculate_frechet_distance(m1, s1, m2, s2)

  mse_avg = mse_total / batch_size
  lpips_loss_avg = lpips_loss_total / batch_size
  sifid_values_avg = sifid_values_total / batch_size
  print('')
  print(sifid_values_avg)
  print(lpips_loss_avg)
  print(mse_avg)
