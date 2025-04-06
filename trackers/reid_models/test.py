import os
import torch
import torch.backends.cudnn as cudnn
import argparse
from tqdm import tqdm
import time

from models import load_model
from utils.datasets import dataloader
from utils.load_yaml import load_yaml

def compute_features(model, test_loader, num_query):
    device = model.device
    model.eval()
    features = torch.tensor([]).float().to(device)
    camera_ids = torch.tensor([]).int().to(device)
    obj_ids = torch.tensor([]).int().to(device)
    
    start = time.time()
    with torch.no_grad():
        for (imgs_batch, cam_ids_batch, pids_batch) in tqdm(test_loader, desc='compute features'):
            imgs_batch = imgs_batch.to(device)
            cam_ids_batch = cam_ids_batch.to(device)
            pids_batch = pids_batch.to(device)

            feats_batch = model(imgs_batch)
            features = torch.cat((features, feats_batch), dim=0)
            camera_ids = torch.cat((camera_ids, cam_ids_batch), dim=0)
            obj_ids = torch.cat((obj_ids, pids_batch))
    inference_time = (time.time() - start)/ len(camera_ids) * 1000
    print(f'{inference_time:.2f} ms per image')
    query_features = features[:num_query]
    query_camera_ids = camera_ids[:num_query]
    query_obj_ids = obj_ids[:num_query]

    gallery_features = features[num_query:]
    gallery_camera_ids = camera_ids[num_query:]
    gallery_obj_ids = obj_ids[num_query:]

    features = {
        "qf": query_features,
        "ql": query_obj_ids,
        "qc": query_camera_ids,
        "gf": gallery_features,
        "gl": gallery_obj_ids,
        "gc": gallery_camera_ids
    }
    return features

def main():

    parser = argparse.ArgumentParser(description='Test ReID model')
    parser.add_argument('--cfg', type=str, default='resnet18.yml',
                        help='configuration file in cfg/')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    cfg = load_yaml(args.cfg)
    data_dir = cfg.data_dir    
    save_folder = cfg.save_folder
    if save_folder is None:
        print('No save folder specified')
        return

    if not torch.cuda.is_available() or cfg.device == 'cpu':
        device = 'cpu'
    elif isinstance(cfg.device, int):
        device = f'cuda:{cfg.device}'
    else:
        device = 'cuda:0'

    if device != 'cpu':
        cudnn.benchmark = True

    print(f'Datapath: {data_dir}')
    print(f'Checkpoint: {save_folder}')
    print(f'image shape: {cfg.image_shape}')
    print(f'Device: {device}')

    # dataloader
    print('Load data ....')
    _, test_loader, num_query = dataloader(
        dir=data_dir,
        image_shape=cfg.image_shape,
        test_batch=cfg.test_batch_size,
        num_workers=4
    )
    parrent_path = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(parrent_path, 'checkpoint', save_folder)


    print('Load checkpoint ....')
    model_path = os.path.join(ckpt_path, 'best_ckpt.pth')
    assert os.path.isfile(model_path), 'Checkpoint not found'
    net = load_model(model_path=model_path, reid=True, feature_dim=cfg.feature_dim)

    # compute features
    features = compute_features(
        model=net,
        test_loader=test_loader,
        num_query=num_query
    )

    # save features (optional)
    if args.save:
        save_path = os.path.join(ckpt_path, 'features.pth')
        torch.save(features, save_path)
        print(f'Features saved to {save_path}')

    # evaluate, optinally
    from evaluate import evaluate
    print('Evaluate features....')
    evaluate(features, metric_distance='cosine')
    # evaluate(features, metric_distance='euclidean')

if __name__ == "__main__":
    main()
