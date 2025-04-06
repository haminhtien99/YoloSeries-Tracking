import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample), ]
        else:
            blocks += [BasicBlock(c_out, c_out), ]
    return nn.Sequential(*blocks)


class ResNet_like(nn.Module):
    @property
    def device(self):
        for p in self.parameters():
            return p.device

    def __init__(self, num_classes=576, reid=False, feature_dim=128, pretrained=False):
        super(ResNet_like, self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # 64 64 32
        self.layer1 = make_layers(64, 64, 2, False)
        # 64 64 32
        self.layer2 = make_layers(64, 128, 2, True)
        # 128 32 16
        self.layer3 = make_layers(128, 256, 2, True)
        # 256 16 8
        self.layer4 = make_layers(256, 512, 2, True)
        # 512 8 4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 512 1 1
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # flatten
        # B x 512
        if self.reid:
            return x
        logits = self.classifier(x)
        return logits, x
    def load_checkpoint(self, weight: None|str):
        model_dict = self.state_dict()
        if weight is None:
            return
        if not os.path.exists(weight):
            reid_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weight = os.path.join(reid_folder, weight)
        try:
            checkpoint = torch.load(weight, map_location='cpu', weights_only=True)
            state_dict = checkpoint['net_dict'] if 'net_dict' in checkpoint else checkpoint
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if k in model_dict and model_dict[k].size() == v.size()
            }
            self.load_state_dict(state_dict, strict=False)
        except FileNotFoundError:
            print(f'Checkpoint not found: {weight}')

if __name__ == '__main__':
    net = ResNet_like()
    x = torch.randn(4, 3, 128, 64)
    y = ResNet_like(x)
