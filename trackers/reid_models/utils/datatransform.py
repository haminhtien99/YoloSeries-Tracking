from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image

class CustomResizePad:
    def __init__(self, target_shape=(128, 128)):
        self.target_height = target_shape[0]
        self.target_width = target_shape[1]
        self.target_aspect = target_shape[0] / target_shape[1]
    def __call__(self, img: Image):
        # Upscale if any dimension is smaller than 32
        if min(img.width, img.height) < 32:
            new_width = img.width * 2
            new_height = img.height * 2
            img = F.resize(
                img,
                [new_height, new_width],
                interpolation=transforms.InterpolationMode.BILINEAR
            )

        aspect_ratio = img.width / img.height
        if aspect_ratio > self.target_aspect:
            pad_width = 0
            pad_height = int(img.height * aspect_ratio/ self.target_aspect) - img.height
        else:
            pad_width = int(img.width * self.target_aspect / aspect_ratio) - img.width
            pad_height = 0
        padding = (
            pad_width // 2,    # Left
            pad_height // 2,   # Top
            pad_width - (pad_width // 2),  # Right
            pad_height - (pad_height // 2) # Bottom
        )
        img = F.pad(img, padding, fill=0)  # Fill with black
        # img = F.pad(img, padding, padding_mode='reflect')
        img = F.resize(
            img,
            [self.target_height, self.target_width],
            interpolation=transforms.InterpolationMode.BILINEAR
        )
        return img

def custom_transform(mode='train', target_shape=(128, 128)):
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize(target_shape),
            transforms.RandomCrop(target_shape, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    if mode != 'train':
        transform = transforms.Compose([
            transforms.Resize(target_shape),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform
