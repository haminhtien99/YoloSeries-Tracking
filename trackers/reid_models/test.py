import os
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torchvision.datasets import ImageFolder
import argparse
from tqdm import tqdm
# from trackers.reid_models.resnet_like import Net
from resnet_like import Net
from resnet import *
from utils.datasets import dataloader
from utils.load_yaml import load_yaml

Nets = {'resnet-like': Net,
        'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50, 'resnet101': resnet101,
        'resnext50_32x4d': resnext50_32x4d, 'resnext101_32x8d': resnext101_32x8d}

def compute_features(model, test_loader, num_query):
    device = model.device
    features = torch.tensor([]).float().to(device)
    camera_ids = torch.tensor([]).int().to(device)
    obj_ids = torch.tensor([]).int().to(device)

    with torch.no_grad():
        for (imgs_batch, cam_ids_batch, pids_batch) in tqdm(test_loader, desc='compute features'):
            imgs_batch = imgs_batch.to(device)
            cam_ids_batch = cam_ids_batch.to(device)
            pids_batch = pids_batch.to(device)

            feats_batch = model(imgs_batch)
            features = torch.cat((features, feats_batch), dim=0)
            camera_ids = torch.cat((camera_ids, cam_ids_batch), dim=0)
            obj_ids = torch.cat((obj_ids, pids_batch))

    query_features = features[:num_query]
    query_cameras = camera_ids[:num_query]
    query_obj_ids = obj_ids[:num_query]

    gallery_features = features[num_query:]
    gallery_cameras = camera_ids[num_query:]
    gallery_obj_ids = obj_ids[num_query:]
    features = {
        "qf": query_features,
        "ql": query_obj_ids,
        "qc": query_cameras,
        "gf": gallery_features,
        "gl": gallery_obj_ids,
        "gc": gallery_cameras
    }
    return features

def main():

    parser = argparse.ArgumentParser(description='Test ReID model')
    parser.add_argument('--config', type=str, default='resnet18.yml',
                        help='configuration file in conf/')
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    data_dir = cfg.data_dir    
    save_folder = cfg.save_folder
    if save_folder is None:
        print('No save folder specified')
        return

    device = 'cuda:{}'.format(cfg.gpu_id) if torch.cuda.is_available() and not cfg.no_cuda else 'cpu'
    if torch.cuda.is_available() and not cfg.no_cuda:
        cudnn.benchmark = True

    print(f'Datapath: {data_dir}')
    print(f'Checkpoint: {save_folder}')
    print(f'image shape: {cfg.image_shape}')
    print(f'Device: {device}')

    # dataloader
    print('Load data ....')
    _, test_loader, num_query = dataloader(data_dir)
    parrent_path = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(parrent_path, 'checkpoint', save_folder)

    
    print('Load checkpoint ....')
    model_path = os.path.join(ckpt_path, 'ckpt.pth')
    assert os.path.isfile(model_path), 'Checkpoint not found'
    net = Nets[cfg.net](reid=True)
    net.load_checkpoint(model_path)
    net.eval()
    net.to(device)

    # compute features
    features = compute_features(
        model=net,
        test_loader=test_loader,
        num_query=num_query
    )
    save_path = os.path.join(ckpt_path, 'features.pth')
    torch.save(features, save_path)
    print(f'Features saved to {save_path}')

    # evaluate, optinally
    from evaluate import evaluate
    evaluate(features, metric_distance='cosine')
    # evaluate(features, metric_distance='euclidean')

if __name__ == "__main__":
    main()
