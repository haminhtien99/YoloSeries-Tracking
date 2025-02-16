import argparse
import os

import numpy as np
import torch
import torchvision
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from model import Net
from utils.datatransform import custom_transform
from utils.plot import plot_results

# train
def train_on_batch(model, x_batch, y_batch, optimizer, loss_function):
    device = model.device
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    model.train()
    optimizer.zero_grad()

    # forward
    output = model(x_batch)
    correct = output.max(dim=1)[1].eq(y_batch).sum().item()
    loss = loss_function(output, y_batch)

    # backward
    loss.backward()
    optimizer.step()

    return loss.cpu().item(), correct

def train_on_epoch(train_generator, optimizer, loss_function, model):
    epoch_loss = 0.
    total = 0
    acc = 0.
    iterations = tqdm(train_generator, desc='Training per epoch')
    iterations.set_postfix({'batch loss': np.nan})
    for (x_batch, y_batch) in iterations:
        batch_loss, corr = train_on_batch(model=model,
                                          x_batch=x_batch,
                                          y_batch=y_batch,
                                          optimizer=optimizer,
                                          loss_function=loss_function)
        epoch_loss += batch_loss * len(y_batch)
        total += len(x_batch)
        acc += corr
        iterations.set_postfix({'batch loss': batch_loss})
    return epoch_loss/total, acc/total

# test during training
def test(loss_function, test_generator, model):
    model.eval()
    device = model.device
    test_loss = 0.
    acc = 0
    total = 0
    with torch.no_grad():
        for (x_batch, y_batch) in test_generator:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = loss_function(output, y_batch)

            test_loss += loss.item() * len(y_batch)
            acc += output.max(dim=1)[1].eq(y_batch).sum().item()
            total += len(x_batch)

    return test_loss/total, acc/total

def checkpoint_setup(model, save_folder, resume=False):

    # Checkpoint location
    parrent_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(parrent_path, 'checkpoint')
    if resume:
        # load history
        full_path = os.path.join(checkpoint_path, save_folder, 'ckpt.pth')
        if os.path.exists(full_path):
            print(f'Loading checkpoint from {full_path}')
            checkpoint = torch.load(full_path, map_location=model.device)
            model.load_state_dict(checkpoint['net_dict'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['acc']
            return best_acc, start_epoch
        else:
            print("Not checkpoint, create new checkpoint")

    # create checkpoint/train.txt
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if save_folder is None:
        save_folder = f'exp{len(os.listdir(checkpoint_path)) + 1}'
    else:
        save_folder = save_folder

    best_acc = 0.
    start_epoch = 0


def trainer(model,
            number_of_epoch,
            train_generator,
            test_generator,
            loss_function,
            optimizer,
            scheduler,
            exp_path: str,
            resume = False):
    best_acc = 0.
    start_epoch = 0
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    full_path = os.path.join(exp_path, 'ckpt.pth')
    if resume:
        # load history
        if os.path.exists(full_path):
            print(f'Loading checkpoint from {full_path}')
            checkpoint = torch.load(full_path, map_location=model.device)
            model.load_state_dict(checkpoint['net_dict'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['acc']
        else:
            print("Not checkpoint")
            return
    else: # create checkpoint/train.txt
        with open(os.path.join(exp_path, 'train.txt'), 'w') as f:
            line = 'epoch,train_loss,test_loss,train_err,test_err\n'
            f.write(line)

    for epoch in range(start_epoch, number_of_epoch + start_epoch):
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch + 1}/{number_of_epoch + start_epoch} lr: {current_lr}')
        train_loss, train_acc = train_on_epoch(train_generator=train_generator,
                                               optimizer=optimizer,
                                               loss_function=loss_function,
                                               model=model)
        scheduler.step()
        # test
        print('Testing ...')
        test_loss, test_acc = test(loss_function=loss_function,
                                   test_generator=test_generator,
                                   model=model)
        print(f'Test loss:{test_loss: .3f}, test_acc:{test_acc: .3f}')
        # saving checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"Saving parameters to {full_path}")
            checkpoint = {
                'net_dict':model.state_dict(),
                'acc':test_acc,
                'epoch':epoch,
            }
            torch.save(checkpoint, full_path)

        # save result training
        with open(os.path.join(exp_path, 'train.txt'), 'a') as f:
            line = f'{epoch + 1},{train_loss},{test_loss},{1. - train_acc},{1. - test_acc}\n'
            f.write(line)

    plot_results(exp_path)
def dataloader(train_dir: str, val_dir: str, image_shape=(128, 128), batch_size=64):
    train_transform = custom_transform(mode='train', target_shape=image_shape)
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(train_dir, transform=train_transform),
        batch_size=batch_size,
        shuffle=True
    )
    val_transform = custom_transform(mode='val', target_shape=image_shape)
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(val_dir, transform=val_transform),
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader

def parser_args():
    parser = argparse.ArgumentParser(description="Train ReID vehicle")
    parser.add_argument("--data-dir",default='/home/ha/Downloads/Dataset/VeRi/pytorch',type=str)
    parser.add_argument("--image-shape", default=(128,64), nargs=2, type=int)
    parser.add_argument("--no-cuda",action="store_true")
    parser.add_argument("--gpu-id",default=0,type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr0",default=0.1, type=float)
    parser.add_argument('--resume', '-r',action='store_true')
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--save-folder", default=None, type=str)
    parser.add_argument("--pretrain", default=None, type=str)

    args = parser.parse_args()
    return args

def main():
    args = parser_args()
    datapath = args.data_dir
    image_shape = args.image_shape
    batch_size = args.batch_size
    # device
    device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda \
        else "cpu"
    if torch.cuda.is_available() and not args.no_cuda:
        cudnn.benchmark = True

    print(f'dataset:{datapath}')
    print(f'image shape:{image_shape}')
    print(f'batch size:{batch_size}')
    print(f'Save to: {save_folder}')
    print(f'device: {device}')
    if args.resume:
        print('Resume training')
    if args.pretrain is not None:
        print(f'Pretrained weight: {args.pretrain}')

    print('--------------------------------')


    # Checkpoint location
    parrent_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(parrent_path, 'checkpoint')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if args.save_folder is None:
        save_folder = f'exp{len(os.listdir(checkpoint_path)) + 1}'
    else:
        save_folder = args.save_folder
    exp_path = os.path.join(checkpoint_path, save_folder)


    # dataloader
    train_dir = os.path.join(datapath,"train")
    val_dir = os.path.join(datapath,"val")
    train_loader, val_loader = dataloader(train_dir=train_dir,
                                          val_dir=val_dir,
                                          image_shape=image_shape,
                                          batch_size=batch_size)

    # net definition
    net = Net(num_classes=len(train_loader.dataset.classes))

    # pretrain weight with VeRi dataset
    if args.pretrain is not None:
        net.load(args.pretrain)
    net.to(device)

    # loss, optimizer and scheduler
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr0, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    trainer(number_of_epoch=args.epochs,
            train_generator=train_loader,
            test_generator=val_loader,
            model=net,
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            exp_path=exp_path,
            resume=args.resume)

if __name__ == '__main__':
    main()
