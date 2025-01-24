import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm


from model import Net
from utils import plot_results, train_loader, test_loader

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir",default='/home/ha/Downloads/Dataset/market-1501/Market-1501-v15.09.15/pytorch',type=str)
parser.add_argument("--no-cuda",action="store_true")
parser.add_argument("--gpu-id",default=0,type=int)
parser.add_argument("--lr0",default=0.1, type=float)
parser.add_argument('--resume', '-r',action='store_true')
parser.add_argument("--epochs", default=3, type=int)
args = parser.parse_args()

# device
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# train
def train_on_batch(model: Net, x_batch, y_batch, optimizer, loss_function):
    model.train()
    optimizer.zero_grad()

    output = model(x_batch.to(device))

    correct = output.max(dim=1)[1].eq(y_batch).sum().item()
    loss = loss_function(output, y_batch.to(device))
    loss.backward()

    optimizer.step()
    return loss.cpu().item(), correct

def train_on_epoch(train_generator, optimizer, loss_function, model, epoch):
    epoch_loss = 0.
    total = 0
    acc = 0.
    iterations = tqdm(train_generator, desc=f'Epoch {epoch}')
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

def trainer(model,
            number_of_epoch,
            train_generator,
            test_generator,
            loss_funtion,
            optim,
            lr0 = 0.1,
            resume = False):
    best_acc = 0.
    optimizer = optim(model.parameters(), momentum=0.9, weight_decay=5e-4, lr=lr0)
    start_epoch = 0
    parrent_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(parrent_path, 'checkpoint')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    full_path = os.path.join(checkpoint_path, 'ckpt.pth')
    if resume:
        # load history
        if os.path.exists(full_path):
            print(f'Loading checkpoint from {full_path}')
            checkpoint = torch.load(full_path)
            model.load_state_dict(checkpoint['net_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['acc']
        else:
            print("Not checkpoint")
            return
    else: # create checkpoint/train.txt
        with open(os.path.join(checkpoint_path, 'train.txt'), 'a') as f:
            line = 'epoch,train_loss,test_loss,train_err,test_err\n'
            f.write(line)

    iterations = tqdm(range(start_epoch, number_of_epoch + start_epoch), desc='Training')
    iterations.set_postfix({'epoch loss': np.nan, 'epoch acc': np.nan})
    for epoch in iterations:
        train_loss, train_acc = train_on_epoch(train_generator=train_generator,
                                               optimizer=optimizer,
                                               loss_function=loss_funtion,
                                               model=model,
                                               epoch=epoch+1)
        iterations.set_postfix({'epoch loss': train_loss, 'epoch acc': train_acc})
        # test
        print("Testing ...")
        test_loss, test_acc = test(loss_function=loss_funtion,
                                              test_generator=test_generator,
                                              model=model)
        print(f'Epoch{epoch+1}: test loss:{test_loss: .3f}, test_acc:{test_acc: .3f}')
        # saving checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"Saving parameters to {full_path}")
            checkpoint = {
                'net_dict':model.state_dict(),
                'acc':test_acc,
                'epoch':epoch,
            }
            if not os.path.isdir(checkpoint_path):
                os.mkdir(checkpoint_path)
            torch.save(checkpoint, full_path)

        # save result training
        with open(os.path.join(checkpoint_path, 'train.txt'), 'a') as f:
            line = f'{epoch + 1},{train_loss},{test_loss},{1. - train_acc},{1. - test_acc}\n'
            f.write(line)

        # learning rate decay
        if (epoch + 1) % 20 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
                print(f'Decay LR to {param_group["lr"]:.6f}')
    plot_results(checkpoint_path)

def main():
    datapath = args.data_dir
    # dataloader
    train_dir = os.path.join(datapath,"train")
    test_dir = os.path.join(datapath,"val")
    trainloader = train_loader(train_dir)
    testloader = test_loader(test_dir)
    num_classes = len(trainloader.dataset.classes)

    # net definition
    net = Net(num_classes=num_classes)
    net.to(device)

    # loss and optimizer
    loss_function = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD
    trainer(number_of_epoch=args.epochs,
            train_generator=trainloader,
            test_generator=testloader,
            model=net,
            loss_funtion=loss_function,
            optim=optim,
            lr0=args.lr0)

if __name__ == '__main__':
    main()
