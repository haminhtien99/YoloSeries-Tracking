import torch
import torchvision
import os
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from torchvision.datasets import ImageFolder
class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, _ = self.samples[index]  # Get image path and label
        image = self.loader(path)  # Load image
        if self.transform:
            image = self.transform(image)
        name = os.path.basename(path)
        name = name.split('_')
        cam_id = int(name[1][1:])
        pid = int(name[0])
        return image, cam_id, pid

def custom_transform(mode='train', target_shape=(128, 128)):
    if mode == 'train':
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
            transforms.Resize(target_shape, interpolation=InterpolationMode.BICUBIC),
            transforms.Pad(10),
            transforms.RandomCrop(target_shape),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(target_shape),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform

def dataloader(dir: str, image_shape=(128, 128), train_batch=64, test_batch=256,
              num_workers=2, pin_memory=True, prefetch_factor=2):
    train_dir = os.path.join(dir, 'train')
    train_transform = custom_transform(mode='train', target_shape=image_shape)
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(train_dir, transform=train_transform),
        batch_size=train_batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor
    )

    test_transform = custom_transform(mode='test', target_shape=image_shape)
    query_dir = os.path.join(dir, 'query')
    gallery_dir = os.path.join(dir, 'gallery')
    query_dataset = CustomImageFolder(query_dir, transform=test_transform)
    gallery_dataset = CustomImageFolder(gallery_dir, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([query_dataset, gallery_dataset]),
        batch_size=test_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor
    )

    return train_loader, test_loader, len(query_dataset)
