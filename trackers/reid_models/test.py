import os
import torch
import torch.backends.cudnn as cudnn
import argparse
from tqdm import tqdm
import time

from models import load_model
from utils.datasets import dataloader
from model_tensorrt import Engine

def compute_features(model, test_loader, num_query, dtype=torch.float32):
    device = model.device
    model.eval()
    features = torch.tensor([]).float().to(device).to(dtype)
    camera_ids = torch.tensor([]).int().to(device)
    obj_ids = torch.tensor([]).int().to(device)

    start = time.time()
    with torch.no_grad():
        for (imgs_batch, cam_ids_batch, pids_batch) in tqdm(test_loader, desc='compute features'):
            imgs_batch = imgs_batch.to(device).to(dtype)
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
    parser.add_argument('--data', type=str,
                        default='/home/ha/Downloads/Dataset/CustomVehicle-ReID/pytorch/')
    parser.add_argument('--ckpt', type=str, help='path to model',
                        default='checkpoint/resnet18-without-triplet-0_0005/best_ckpt.pth')
    parser.add_argument('--imgsz', type=int, nargs=2, default=[128, 128],
                        help='image size (height width)')
    parser.add_argument('--half', action='store_true', help='half for evaluating')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--feature_dim', type=int, default=128,
                        help='output feature dim descriptor')
    parser.add_argument('--batch', type=int, default=256,
                        help='test batch size')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    data = args.data
    ckpt = args.ckpt

    if not torch.cuda.is_available() or args.device == 'cpu':
        device = 'cpu'
    elif isinstance(args.device, int):
        device = f'cuda:{args.device}'
    else:
        device = 'cuda:0'

    if device != 'cpu':
        cudnn.benchmark = True

    print(f'Datapath: {data}')
    print(f'Checkpoint: {ckpt}')
    print(f'image shape: {args.imgsz}')
    print(f'Device: {device}')

    # dataloader
    print('Load data ....')
    _, test_loader, num_query = dataloader(
        dir=data,
        image_shape=args.imgsz,
        test_batch=args.batch,
        num_workers=4,
        pin_memory=False if device == 'cpu' else True
    )


    print('Load checkpoint ....')
    assert os.path.isfile(ckpt), 'Checkpoint not found'
    format = ckpt.split('.')[-1]

    if format in ['pth', 'pt']:
        net = load_model(model_path=ckpt, reid=True, feature_dim=args.feature_dim)
        dtype = torch.float16 if args.half else torch.float32
        net.to(dtype)
    elif format in ['engine', 'trt']:
        net = Engine(model_path=ckpt, device=device)
        dtype = torch.float16 if net.fp16 else torch.float32

    else:
        raise ValueError(f"Unsupported model format: {format}")
    print(dtype)

    # compute features
    features = compute_features(
        model=net,
        test_loader=test_loader,
        num_query=num_query,
        dtype=dtype
    )

    # save features (optional)
    if args.save:
        save_folder = os.path.basename(ckpt)
        save_path = os.path.join(save_folder, 'features.pth')
        torch.save(features, save_path)
        print(f'Features saved to {save_path}')

    # evaluate, optinally
    from evaluate import evaluate
    print('Evaluate features....')
    evaluate(features, metric_distance='cosine')

if __name__ == "__main__":
    main()
