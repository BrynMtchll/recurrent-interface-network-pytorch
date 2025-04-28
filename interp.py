import torch.nn.functional as F
import blobfile as bf
from PIL import Image
from torchvision import transforms as T, utils
from torchvision import transforms as T, utils

resolution = 249

with bf.BlobFile('/Users/brynly/Documents/courses/engn4528/recurrent-interface-network-pytorch/data/comp4528-mini-proj-image.png', "rb") as f:
    target_img = Image.open(f)
    target_img = target_img.convert("RGB")
    target_img.load()

transform = T.Compose([
    T.Resize(resolution),
    T.CenterCrop(resolution),
    T.ToTensor()
])
target_img = transform(target_img)

utils.save_image(target_img, f'/Users/brynly/Documents/courses/engn4528/recurrent-interface-network-pytorch/interp/source-{resolution}.png')


# Open an image
img = Image.open(f'/Users/brynly/Documents/courses/engn4528/recurrent-interface-network-pytorch/interp/source-{resolution}.png')

# Resize with bicubic interpolation
img_resized = img.resize((1024, 1024), Image.BICUBIC)

# Save
img_resized.save(f'/Users/brynly/Documents/courses/engn4528/recurrent-interface-network-pytorch/interp/interp-1024.png')