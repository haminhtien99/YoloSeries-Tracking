import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F 
import torch.backends.cudnn as cudnn
from pytorch_metric_learning import losses, miners
from tqdm import tqdm


# from trackers.reid_models.resnet_like import Net
# from trackers.reid_models.resnet import *
from resnet_like import Net
from resnet import *
from evaluate import build_dist, evaluate_rank
from utils.datasets import dataloader
from utils.log import prepare_training, save_checkpoint
from utils.lr_scheduler import fastReID_lr_lambda
from utils.load_yaml import load_yaml
Nets = {'resnet-like': Net,
        'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50, 'resnet101': resnet101,
        'resnext50_32x4d': resnext50_32x4d, 'resnext101_32x8d': resnext101_32x8d}

# train
def train_on_batch(model, x_batch, y_batch, optimizer, criterion, metric_loss=None, miner=None):
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

def train_on_epoch(train_generator, optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR,
                   criterion, model,
                   metric_loss=None, miner=None, freeze=False):
    epoch_loss = 0.
    total = 0
    acc = 0.
    process_bar = tqdm(train_generator, desc='Training per epoch')
    process_bar.set_postfix({'batch loss': np.nan, 'iter': scheduler.last_epoch, 'lr': scheduler.get_last_lr()})
    for (x_batch, y_batch) in process_bar:
        batch_loss, corr = train_on_batch(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            optimizer=optimizer,
            criterion=criterion,
            metric_loss=metric_loss,
            miner=miner
        )
        scheduler.step()
        epoch_loss += batch_loss * len(y_batch)
        total += len(x_batch)
        acc += corr
        process_bar.set_postfix({'batch loss': batch_loss, 'iter': scheduler.last_epoch, 'lr': scheduler.get_last_lr()[0]})

        # unfreeze backbone model and reinitialize scheduler
        iteration = scheduler.last_epoch
        if freeze:
            if iteration == 2000:
                for name, param in model.named_parameters():
                    if 'fc' not in name:
                        param.requires_grad = True
                optimizer.add_param_group({
                    'params': [p for p in model.parameters() if 'fc' not in p],
                    'lr': 3.5e-4
                })
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fastReID_lr_lambda, last_epoch=iteration)
            freeze = False
    return epoch_loss/total, acc/total

def test(test_loader, model, num_query, metric_distance='cosine'):
    model.eval()
    device = model.device
    features = torch.tensor([]).float().to(device)
    camera_ids = torch.tensor([]).int().to(device)
    pids = torch.tensor([]).int().to(device)
    with torch.no_grad():
        for (imgs_batch, cam_ids_batch, pids_batch) in test_loader:
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
        scheduler,
        exp_path: str,
        metric_loss=None,
        miner=None,
        resume=False,
        freeze=False
    ):
    best_metric, start_epoch = prepare_training(resume, model, optimizer, scheduler, exp_path)
    for epoch in range(start_epoch, number_of_epoch + start_epoch):
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch + 1}/{number_of_epoch + start_epoch} lr: {current_lr}')
        train_loss, train_acc = train_on_epoch(
            train_generator=train_generator,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            metric_loss=metric_loss,
            miner=miner,
            model=model,
            freeze=freeze)

        # test
        print('Testing ...')
        test_results = test(
            test_loader=test_generator,
            model=model,
            num_query=num_query,
            metric_distance='cosine'
        )
        r1, mAP, mINP = test_results[0], test_results[-2], test_results[-1]
        print(f'Rank@1:{r1:.2f}, mAP:{mAP:.2f}, mINP:{mINP:.2f}')
        # saving checkpoint
        save_checkpoint(
            epoch, model, optimizer, scheduler, test_results,
            [train_loss, train_acc], exp_path, best_metric
        )

def main():
    parser = argparse.ArgumentParser(description='Train ReID vehicle')
    parser.add_argument('--config', type=str, default='resnet18.yml',
                        help='configuration file in conf/')
    args = parser.parse_args()
    config = load_yaml(args.config)
    datapath = config.data_dir
    image_shape = config.image_shape
    # device
    device = "cuda:{}".format(config.gpu_id) if torch.cuda.is_available() and not config.no_cuda \
        else "cpu"
    if torch.cuda.is_available() and not config.no_cuda:
        cudnn.benchmark = True

    print(f'dataset:{datapath}')
    print(f'net: {config.net}')
    print(f'image shape:{image_shape}')
    print(f'device: {device}')
    if config.resume:
        print('Resume training')
    if config.pretrained:
        print('Load pretrained model')

    print('--------------------------------')

    # Checkpoint location
    parrent_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(parrent_path, 'checkpoint')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if config.save_folder is None:
        save_folder = f'exp{len(os.listdir(checkpoint_path)) + 1}'
    else:
        save_folder = config.save_folder
    exp_path = os.path.join(checkpoint_path, save_folder)


    # dataloader
    train_loader, test_loader, num_query = dataloader(
        datapath,
        image_shape=image_shape,
        train_batch=config.train_batch_size,
        test_batch=config.test_batch_size,
    )

    # net definition
    if config.net == 'resnet-like':
        net = Net(num_classes=len(train_loader.dataset.classes))
    else:
        # Using pretrained weights doesn't really work for the VeRi dataset
        net = Nets[config.net](num_classes=len(train_loader.dataset.classes),
                             pretrained=config.pretrained)

    net.to(device)

    # loss, optimizer and scheduler
    criterion = torch.nn.CrossEntropyLoss()
    miner = miners.MultiSimilarityMiner()
    metric_loss = losses.TripletMarginLoss(margin=0.3)
    if config.pretrained: # work very badly
        # freeze backbone reid model
        optimizer = torch.optim.Adam(net.fc.parameters(), lr=3.5e-6)
        for name, param in net.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=fastReID_lr_lambda)
        freeze = True

    else:
        # optimizer = torch.optim.SGD(net.parameters(), lr=config.lr0, momentum=0.9, weight_decay=5e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)

        optimizer = torch.optim.SGD(net.parameters(), lr=config.lr0, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.lr0,                 # Peak learning rate
            total_steps=len(train_loader) * config.epochs,  # Total batches across all epochs
            pct_start=0.3,              # Warmup phase (30% of iterations)
            anneal_strategy='cos',      # Cosine annealing
        )
        freeze = False

    trainer(number_of_epoch=config.epochs,
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
            resume=config.resume,
            freeze=freeze)

if __name__ == '__main__':
    main()
