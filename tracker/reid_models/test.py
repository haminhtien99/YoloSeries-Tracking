import os
import torch
import torch.backends.cudnn as cudnn
import torchvision
import argparse
from tqdm import tqdm
from model import Net
from utils.datatransform import custom_transform

def compute_features(model, queryloader, galleryloader):
    # compute features
    query_features = torch.tensor([]).float()
    query_labels = torch.tensor([]).long()
    gallery_features = torch.tensor([]).float()
    gallery_labels = torch.tensor([]).long()
    device = model.device
    with torch.no_grad():
        for (inputs,labels) in tqdm(queryloader, desc='compute query features -----'):
            inputs = inputs.to(device)
            features = model(inputs).cpu()
            query_features = torch.cat((query_features, features), dim=0)
            query_labels = torch.cat((query_labels, labels))

        for (inputs,labels) in tqdm(galleryloader, desc='compute gallery features ---'):
            inputs = inputs.to(device)
            features = model(inputs).cpu()
            gallery_features = torch.cat((gallery_features, features), dim=0)
            gallery_labels = torch.cat((gallery_labels, labels))

    features = {
    "qf": query_features,
    "ql": query_labels,
    "gf": gallery_features,
    "gl": gallery_labels
    }
    return features

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
    parser = argparse.ArgumentParser(description='Evaluate ReID model')
    parser.add_argument('--datadir', default='/home/ha/Downloads/Dataset/VeRi/pytorch',
                        type=str)
    parser.add_argument('--ckpt-folder', default='exp1-128-128', type=str,
                        help='The checkpoint folder with weight')
    parser.add_argument("--no-cuda",action="store_true")
    parser.add_argument("--gpu-id",default=0,type=int)
    args = parser.parse_args()
    return args

def main(args):

    print(f'Datapath: {args.datadir}')
    print(f'Checkpoint: {args.ckpt_folder}')

    device = 'cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    if torch.cuda.is_available() and not args.no_cuda:
        cudnn.benchmark = True    
    print(f'Device: {device}')

    # dataloader
    print('Load data ....')
    datadir = args.datadir
    query_dir = os.path.join(datadir, 'query')
    gallery_dir = os.path.join(datadir, 'gallery')
    transform = custom_transform(mode='test', target_shape=(128, 128))
    queryloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(query_dir, transform=transform),
        batch_size=64,
        shuffle=True
    )
    galleryloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(gallery_dir, transform=transform),
        batch_size=64,
        shuffle=False
    )

    # load model
    parrent_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(parrent_path, 'checkpoint', args.ckpt_folder, 'ckpt.pth')
    assert os.path.isfile(model_path), 'Checkpoint not found'
    print('Load checkpoint ....')
    net = Net(reid=True)
    net.load(model_path)
    net.eval()
    net.to(device)

    # compute features and evaluate
    features = compute_features(model=net,
                                galleryloader=galleryloader,
                                queryloader=queryloader)
    res = evaluate(features)
    print(f'Accuracy: Rank@1 {res: .3f}')

if __name__ == "__main__":
    args = parser_args()
    main(args)
