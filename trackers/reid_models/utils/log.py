import os
import matplotlib.pyplot as plt
import torch
import pandas as pd

def plot_results(outpath: str):
    txt_path = os.path.join(outpath, 'train.txt')
    if not os.path.exists(txt_path):
        print(f"Results file {txt_path} not found!")
        return

    # Load data with error handling
    try:
        df = pd.read_csv(txt_path)
    except Exception as e:
        print(f"Error loading results: {e}")
        return

    # Create plots with proper layout
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training Metrics", fontsize=14)
    
    # Plot configuration
    metrics = [
        ('train_loss', 'Training Loss'),
        ('train_acc', 'Training Accuracy'),
        ('r1', 'Rank-1 Accuracy'),
        ('mAP', 'mAP'),
        ('mINP', 'mINP')
    ]
    
    for idx, (col, title) in enumerate(metrics):
        ax = axs.flatten()[idx]
        if col in df.columns:
            ax.plot(df['epoch'], df[col], 'o-', label=title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.grid(True)
        else:
            ax.axis('off')  # Hide empty subplots
    
    # Remove empty subplot
    if len(metrics) < 6:
        axs[-1, -1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, 'train.jpg'))
    plt.close()


def save_checkpoint(epoch, model, optimizer, scheduler, test_results, train_results, exp_path, best_metric):
    # best_metric is Rank@1
    if best_metric > test_results[0]:
        best_metric = test_results[0]

    checkpoint = {
        'epoch': epoch,
        'net_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'scheduler_dict': scheduler.state_dict(),
        'best_metric': best_metric,
    }
    torch.save(checkpoint, os.path.join(exp_path, 'ckpt.pth'))

    # save to train.txt
    train_loss, train_acc = train_results
    r1, r5, r10, mAP, mINP = test_results
    with open(os.path.join(exp_path, 'train.txt'), 'a') as f:
        line = f'{epoch},{train_loss},{train_acc},{r1},{r5},{r10},{mAP},{mINP}'
        f.write(line)

    # plot result
    plot_results(exp_path)

def prepare_training(resume, model: torch.nn.Module, optimizer, scheduler, exp_path):
    best_metric = 0.
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
            best_metric = checkpoint['best_metric']
            optimizer.load_state_dict(checkpoint['optimmizer_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
        else:
            print("Not checkpoint")
            return
    else:
        # create checkpoint/train.txt
        with open(os.path.join(exp_path, 'train.txt'), 'w') as f:
            line = 'epoch,train_loss,train_err,test_r1,test_r5,test_r10,test_mAP,test_mINP\n'
            f.write(line)
    return best_metric, start_epoch
