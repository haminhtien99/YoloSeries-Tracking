import os
import torch
import torch.backends.cudnn as cudnn
import argparse
from .model import Net
from .utils import test_loader

def evaluate(features):
    qf = features['qf']
    ql = features['ql']
    gf = features['gf']
    gl = features['gl']
    scores = qf.mm(gf.t())
    res = scores.topk(5, dim=1)[1][:, 0]
    top1correct = gl[res].eq(ql).sum().item()
    return top1correct / ql.size(0)

def parser_args():
    parser = argparse.ArgumentParser(description='Train on VeRi')
    parser.add_argument('--datadir', default='', type=str)
    parser.add_argument("--no-cuda",action="store_true")
    parser.add_argument("--gpu-id",default=0,type=int)
    args = parser.parse_args()
    return args
def main(args):
    device = 'cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    if torch.cuda.is_available() and not args.no_cuda:
        cudnn.benchmark = True
    
    datadir = args.datadir
    query_dir = os.path.join(datadir, 'query')
    gallery_dir = os.path.join(datadir, 'gallery')
    queryloader = test_loader(query_dir)
    galleryloader = test_loader(gallery_dir)

    # load model
    net = Net(reid=True)
    assert os.path.isfile('checkpoint/ckpt.pth'), 'Checkpoint not found'
    print('Load checkpoint')
    checkpoint = torch.load('checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net_dict'], strict=False)
    net.eval()
    net.to(device)

    # compute features
    query_features = torch.tensor([]).float()
    query_labels = torch.tensor([]).long()
    gallery_features = torch.tensor([]).float()
    gallery_labels = torch.tensor([]).long()

    with torch.no_grad():
        for idx,(inputs,labels) in enumerate(queryloader):
            inputs = inputs.to(device)
            features = net(inputs).cpu()
            query_features = torch.cat((query_features, features), dim=0)
            query_labels = torch.cat((query_labels, labels))

        for idx,(inputs,labels) in enumerate(galleryloader):
            inputs = inputs.to(device)
            features = net(inputs).cpu()
            gallery_features = torch.cat((gallery_features, features), dim=0)
            gallery_labels = torch.cat((gallery_labels, labels))
    gallery_labels -= 2
    features = {
    "qf": query_features,
    "ql": query_labels,
    "gf": gallery_features,
    "gl": gallery_labels
    }
    res = evaluate(features)
    print(f'Accuracy: Rank@1 {res: .3f}')

if __name__ == "__main__":
    args = parser_args()
    main(args)