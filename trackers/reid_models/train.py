import argparse
import os
import gc
import numpy as np
import torch
import time
import torch.nn.functional as F 
import torch.backends.cudnn as cudnn
from pytorch_metric_learning import losses, miners
from tqdm import tqdm


# from trackers.reid_models.resnet_like import Net
# from trackers.reid_models.resnet import *

from models import *

from evaluate import build_dist, evaluate_rank
from utils.datasets import dataloader
from utils.log import prepare_training, save_checkpoint
from utils.lr_scheduler import fastReID_lr_lambda
from utils.lr_scheduler import build_lr_scheduler
from utils.load_yaml import load_yaml

Nets = {'resnet-like': ResNet_like,
        'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50, 'resnet101': resnet101,
        'osnet_x1_0': osnet_x1_0, 'osnet_x0_75': osnet_x0_75,
        'osnet_x0_5': osnet_x0_5, 'osnet_x0_25': osnet_x0_25,
        'osnet_ibn_x1_0': osnet_ibn_x1_0}

# train
def train_on_batch(
        model,
        x_batch, y_batch,
        optimizer, criterion, metric_loss=None, miner=None,
    ):
    device = model.device
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    model.train()
    optimizer.zero_grad()

    # forward
    logits, features = model(x_batch)
    features = features.div(features.norm(p=2, dim=1, keepdim=True))

    correct = logits.max(dim=1)[1].eq(y_batch).sum().item()
    loss = criterion(logits, y_batch)
    if miner is not None and metric_loss is not None:
        hard_pairs = miner(features, y_batch)
        loss += metric_loss(features, y_batch, hard_pairs)

    # backward
    loss.backward()
    optimizer.step()

    return loss.cpu().item(), correct

def train_on_epoch(
        epoch, epochs, train_generator,
        optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR,
        criterion, model,
        metric_loss=None, miner=None):
    epoch_loss = 0.
    total = 0
    acc = 0.
    print(f"{'Epoch':>11}{'GPU':>11}{'loss':>11}{'lr':>11}")
    pbar = tqdm(train_generator)
    for (x_batch, y_batch) in pbar:
        batch_loss, corr = train_on_batch(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            optimizer=optimizer,
            criterion=criterion,
            metric_loss=metric_loss,
            miner=miner
        )
        device = model.device

        pbar.set_description(
            ("%11s"*2 + "%11.4g" * 2)
            % (
                f"{epoch+1}/{epochs}",
                f"{get_memory(device):>10.3g}G",
                batch_loss,
                scheduler.get_last_lr()[0]
            )
        )
        epoch_loss += batch_loss * len(y_batch)
        total += len(x_batch)
        acc += corr
    scheduler.step()
    clear_memory(device)
    return epoch_loss/total, acc/total

def test(test_loader, model, num_query, metric_distance='cosine'):
    model.eval()
    device = model.device
    features = torch.tensor([]).float().to(device)
    camera_ids = torch.tensor([]).int().to(device)
    pids = torch.tensor([]).int().to(device)
    pbar = tqdm(test_loader, desc=f"{'Time':>11}{'Rank@1':>11}{'mAP':>11}{'mINP':>11}")
    with torch.no_grad():
        for (imgs_batch, cam_ids_batch, pids_batch) in pbar:
            imgs_batch = imgs_batch.to(device)
            cam_ids_batch = cam_ids_batch.to(device)
            pids_batch = pids_batch.to(device)

            _, feats_batch = model(imgs_batch)
            features = torch.cat((features, feats_batch), dim=0)
            camera_ids = torch.cat((camera_ids, cam_ids_batch), dim=0)
            pids = torch.cat((pids, pids_batch))

    query_features = features[:num_query]
    query_camera_ids = camera_ids[:num_query]
    query_pids = pids[:num_query]

    gallery_features = features[num_query:]
    gallery_camera_ids = camera_ids[num_query:]
    gallery_pids = pids[num_query:]

    dist = build_dist(query_features, gallery_features, metric_distance=metric_distance)
    cmc, all_AP, all_INP = evaluate_rank(
        dist,
        query_pids.cpu().numpy(), gallery_pids.cpu().numpy(),
        query_camera_ids.cpu().numpy(), gallery_camera_ids.cpu().numpy(),
        max_rank=50
    )
    mAP = np.mean(all_AP) * 100
    mINP = np.mean(all_INP) * 100
    rank1 = cmc[0] * 100
    rank5 = cmc[4] * 100
    rank10 = cmc[9] * 100
    return [rank1, rank5, rank10, mAP, mINP]

def trainer(
        model,
        number_of_epoch,
        train_generator,
        test_generator,
        num_query,
        criterion,
        optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        exp_path: str,
        early_stop=10,
        metric_loss=None,
        miner=None,
        resume=False
    ):
    time_from_update = 0
    best_metric, start_epoch = prepare_training(resume, model, optimizer, scheduler, exp_path)
    for epoch in range(start_epoch, number_of_epoch):
        train_loss, train_acc = train_on_epoch(
            train_generator=train_generator,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            metric_loss=metric_loss,
            miner=miner,
            model=model,
            epoch=epoch,
            epochs=number_of_epoch
        )

        # test
        start = time.time()
        test_results = test(
            test_loader=test_generator,
            model=model,
            num_query=num_query,
            metric_distance='cosine'
        )
        end = time.time()
        r1, mAP, mINP = test_results[0], test_results[3], test_results[4]
        print(f"{(end - start):>11.2f}{r1:>11.2f}{mAP:>11.2f}{mINP:>11.2f}")
        # saving checkpoint
        best_metric, time_from_update = save_checkpoint(
            epoch, model, optimizer, scheduler, test_results,
            [train_loss, train_acc], exp_path, best_metric,
            time_from_update=time_from_update
        )
        if time_from_update >= early_stop:
            print('Early stopping at epoch', epoch)
            break

def get_memory(device):
    if device == 'cpu':
        memory = 0
    else:
        memory = torch.cuda.memory_reserved()
    return memory/1e9
def clear_memory(device='cpu'):
    gc.collect()
    if device == 'cpu':
        return
    else:
        torch.cuda.empty_cache()
def main():
    parser = argparse.ArgumentParser(description='Train ReID vehicle')
    parser.add_argument('--cfg', type=str, default='resnet18.yaml',
                        help='configuration file in conf/')
    args = parser.parse_args()
    cfg = load_yaml(args.cfg)
    datapath = cfg.data_dir
    image_shape = cfg.image_shape
    # device
    device = "cuda:{}".format(cfg.gpu_id) if torch.cuda.is_available() and not cfg.no_cuda \
        else "cpu"
    if torch.cuda.is_available() and not cfg.no_cuda:
        cudnn.benchmark = True

    print(f'dataset:{datapath}')
    print(f'net: {cfg.net}')
    print(f'image shape:{image_shape}')
    print(f'device: {device}')
    if cfg.resume:
        print('Resume training')
    print('--------------------------------')

    # Checkpoint location
    parrent_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(parrent_path, 'checkpoint')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if cfg.save_folder is None:
        save_folder = f'exp{len(os.listdir(checkpoint_path)) + 1}'
    else:
        save_folder = cfg.save_folder
    exp_path = os.path.join(checkpoint_path, save_folder)


    # dataloader
    train_loader, test_loader, num_query = dataloader(
        datapath,
        image_shape=image_shape,
        train_batch=cfg.train_batch_size,
        test_batch=cfg.test_batch_size,
        num_workers=4
    )

    # net definition
    net = Nets[cfg.net](num_classes=len(train_loader.dataset.classes))

    net.to(device)

    # loss, optimizer and scheduler
    criterion = torch.nn.CrossEntropyLoss()
    miner = miners.MultiSimilarityMiner()
    metric_loss = losses.TripletMarginLoss(margin=0.3)

    if cfg.optim == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
    elif cfg.optim == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=5e-4)

    scheduler = build_lr_scheduler(optimizer, warmup=0, epochs=50, type_scheduler=cfg.scheduler)

    trainer(number_of_epoch=cfg.epochs,
            train_generator=train_loader,
            test_generator=test_loader,
            num_query=num_query,
            model=net,
            criterion=criterion,
            metric_loss=metric_loss,
            miner=miner,
            optimizer=optimizer,
            scheduler=scheduler,
            exp_path=exp_path,
            resume=cfg.resume)

if __name__ == '__main__':
    main()
