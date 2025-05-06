from .osnet import *
from .resnet_like import ResNet_like
from .resnet import *
from .resnet_ibn import *
from .original_deepsort_model import DeepSortReID
# __all__ = 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'ResNet_like',\
# 'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 'osnet_ibn_x1_0',\
# 'ResNet_IBN', 'resnet18_ibn_a', 'resnet34_ibn_a', 'resnet50_ibn_a', 'resnet101_ibn_a', 'resnet152_ibn_a',\
# 'resnet18_ibn_b', 'resnet34_ibn_b', 'resnet50_ibn_b', 'resnet101_ibn_b', 'resnet152_ibn_b'

Nets = {'resnet-like': ResNet_like, 'deepsort-reid': DeepSortReID,
        'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50, 'resnet101': resnet101,
        'osnet_x1_0': osnet_x1_0, 'osnet_x0_75': osnet_x0_75,
        'osnet_x0_5': osnet_x0_5, 'osnet_x0_25': osnet_x0_25,
        'osnet_ibn_x1_0': osnet_ibn_x1_0}
def load_model(model_path: str, reid=False, weights_only=True, feature_dim=128, num_classes=1000):
    import torch
    ckpt = torch.load(model_path, map_location='cpu', weights_only=weights_only)
    state_dict = ckpt['net_dict']
    name = ckpt['name']
    model: torch.nn.Module = Nets[name](reid=reid, pretrained=False, feature_dim=feature_dim, num_classes=num_classes)
    model_dict = model.state_dict()
    matched_state_dict = {
        k: v
        for k, v in state_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    num_params_loaded = len(matched_state_dict)
    total_params_in_ckpt = len(state_dict)
    print(f"Loaded {num_params_loaded}/{total_params_in_ckpt} parameters from {model_path}.")
    model.load_state_dict(matched_state_dict, strict=False)
    return model
