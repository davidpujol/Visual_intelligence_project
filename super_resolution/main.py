from torchsr.models import edsr_r32f256
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import transforms as transforms
import einops

from torchsr.datasets import Div2K

# Div2K dataset
#dataset = Div2K(root="./data", scale=2, download=False)

# Get the first image in the dataset (High-Res and Low-Res)
#hr, lr = dataset[0]
#print(lr)
#sys.exit()


#image_path = '../human_body_generation/test_inference_img/image2.jpg'
image_path = '../human_body_generation/test_inference_img/butterfly.png'
#image_path = './data/DIV2K/DIV2K_train_LR_bicubic/X2/0001x2.png'
image = Image.open(image_path).convert('RGB')

image = to_tensor(image)

# Plot the original image
image2 = einops.rearrange(image, 'c h w -> h w c')
plt.imshow(image2)
plt.show()

# add a batch dimension
image = image.unsqueeze(0)

# Load the model
model = edsr_r32f256(scale=2, pretrained=True)

# Run the Super-Resolution model
sr_image = model(image)
print(sr_image.shape)
sr = to_pil_image(sr_image.squeeze(0))
sr.show()