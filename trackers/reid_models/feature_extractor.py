import torch
from torchvision import transforms
import numpy as np
import cv2
from .model import Net


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.net.load(model_path)
        self.net.to(self.device)
        self.size = (128, 128)
        self.norm = transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _preprocess(self, im_crops):
        def _resize(im, size):
            return cv2.resize(im, size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch
    def __call__(self, im_crops):

        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()

if __name__ == '__main__':
    import os
    img = cv2.imread(os.path.join(os.path.dirname(__name__), 'demo.jpg'))[:,:,(2,1,0)] # BGR to RGB
    extr = Extractor("checkpoint/visdrone-128-128/ckpt.pth")
    feature = extr([img])
    print(feature.shape)
