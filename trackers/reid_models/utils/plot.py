import os
import matplotlib.pyplot as plt

def plot_results(outpath: str):
    # load results
    txt = os.path.join(outpath, 'train.txt')
    if not os.path.exists(txt):
        return
    epochs, train_loss, test_loss, train_err, test_err = [], [], [], [], []
    with open(txt, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip().split(',')
            epochs.append(int(line[0]))
            train_loss.append(float(line[1]))
            test_loss.append(float(line[2]))
            train_err.append(float(line[3]))
            test_err.append(float(line[4]))

    # plot results
    res_jpg = os.path.join(outpath, 'train.jpg')
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1err")
    ax0.plot(epochs, train_loss, 'bo-', label='train')
    ax0.plot(epochs, test_loss, 'ro-', label='val')
    ax1.plot(epochs, train_err, 'bo-', label='train')
    ax1.plot(epochs, test_err, 'ro-', label='val')
    ax0.legend()
    ax1.legend()
    fig.savefig(res_jpg)
