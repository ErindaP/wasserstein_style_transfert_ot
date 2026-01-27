import torch
from PIL import Image
import torchvision.transforms.functional as TF

def load_image(path, size=None):
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize((size, size))
    # Transform to tensor
    img_t = TF.to_tensor(img).unsqueeze(0)
    return img_t

def save_image(tensor, path):
    # tensor is expected to be (1, C, H, W) or (C, H, W)
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # clamp values
    tensor = torch.clamp(tensor, 0, 1)
    
    img = TF.to_pil_image(tensor)
    img.save(path)
