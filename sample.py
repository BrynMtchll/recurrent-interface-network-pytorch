from torchvision import transforms as T, utils
import torch
from train import diffusion

sampled_images = diffusion.sample(image_size= 128, batch_size = 4)
# images_cat = torch.cat(sampled_images, dim = 0)
print(sampled_images.shape)
utils.save_image(sampled_images, '/content/sample.png')