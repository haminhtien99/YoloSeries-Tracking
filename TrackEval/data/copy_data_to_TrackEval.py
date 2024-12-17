import os
import shutil
from tqdm import tqdm
import argparse
DEST_FOLDER = '/home/ha/projects/YoloSeries-Tracking/TrackEval/data/gt'
def main(args):
    BENCHMARK = args.BENCHMARK
    new_folder = os.path.join(DEST_FOLDER, BENCHMARK)
    if not os.path.exists(new_folder):
        print(f'Folder {new_folder} not exists, create a new one')
        os.makedirs(new_folder)
    else:
        print(f'Folder {new_folder} already exists, remove it and create a new one')
        shutil.rmtree(new_folder)
        os.makedirs(new_folder)

    mot_path = args.mot_path
    splits_set = os.listdir(mot_path)
    for spl_set in splits_set:
        if 'train' in spl_set:
            spl_folder = os.path.join(new_folder, f'{BENCHMARK}-train')
        elif 'test' in spl_set:
            spl_folder = os.path.join(new_folder, f'{BENCHMARK}-test')
        elif 'val' in spl_set:
            spl_folder = os.path.join(new_folder, f'{BENCHMARK}-val')
        else:
            assert False, 'your dataset have to be train/val/test'
        os.makedirs(spl_folder)
        # copy gt files and seqinfo.ini to TrackEval/data/gt
        videos = os.listdir(os.path.join(mot_path, spl_set))
        videos.sort()
        for video in tqdm(videos, desc=f'{BENCHMARK}-{spl_set}'):
            src_gt = os.path.join(mot_path, spl_set, video, 'gt_mot_challenge', 'gt.txt')
            src_seqinfo = os.path.join(mot_path, spl_set, video, 'seqinfo.ini')
            os.makedirs(os.path.join(spl_folder, video, 'gt'))
            shutil.copy2(src_gt, os.path.join(spl_folder, video, 'gt', 'gt.txt'))
            shutil.copy2(src_seqinfo, os.path.join(spl_folder, video, 'seqinfo.ini'))

        # create seqmap
        seqmaps = os.path.join(new_folder, 'seqmaps')
        if not os.path.exists(seqmaps):
            os.makedirs(seqmaps)
        seqmap_name = spl_folder.split('/')[-1]
        seqmap_path = os.path.join(seqmaps, f'{seqmap_name}.txt')
        lines = ['name\n']
        for video in videos:
            lines.append(f'{video}\n')
        with open(seqmap_path, 'w') as f:
            f.writelines(lines)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='copy data to TrackEval')
    parser.add_argument('--BENCHMARK', type=str, help='mot dataset name')
    parser.add_argument('--mot-path', type=str, help='path to mot dataset')
    args = parser.parse_args()
    main(args)