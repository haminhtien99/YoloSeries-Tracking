# UAVDT not have ground truth for some frames.
# Relabel the dataset to make it complete.

import os
from tqdm import tqdm
import shutil
import cv2
DATAPATH = '/home/ha/Downloads/Dataset/UAV-benchmark-M'
NEW_PATH = '/home/ha/Downloads/Dataset/UAV-benchmark-M_relabel'
GT_PATH = '/home/ha/Downloads/Dataset/UAV-benchmark-MOTD_v1.0/GT'
if os.path.exists(NEW_PATH):
    shutil.rmtree(NEW_PATH)
    os.makedirs(NEW_PATH)
else: os.makedirs(NEW_PATH)

videos = os.listdir(DATAPATH)
for v in tqdm(videos, desc='copy videos to new dataset'):
    v_path = os.path.join(DATAPATH, v)
    new_v_path = os.path.join(NEW_PATH, v)
    shutil.copytree(v_path, new_v_path)
gt_files = [gt for gt in os.listdir(GT_PATH) if gt.endswith('gt_whole.txt')]
gt_files.sort()
for gt in tqdm(gt_files, desc='create new gt files'):
    v = gt.split('_')[0]
    gt_folder = os.path.join(NEW_PATH, v + '_gt')
    if not os.path.exists(gt_folder):
        os.makedirs(gt_folder)
    with open(os.path.join(GT_PATH, gt), 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(',')
            frame_id = int(line[0])
            track_id = line[1]
            x, y, w, h = line[2:6]
            obj_id = line[-1]
            new_gt = os.path.join(gt_folder, f'img{frame_id:06d}.txt')
            with open(new_gt, 'a') as another_file:
                another_file.write(f'{obj_id},{x},{y},{w},{h},{track_id}\n')
cls = {1: 'car', 2: 'truck', 3: 'bus'}
for v in tqdm(videos, desc='draw bbox'):
    for gt in os.listdir(os.path.join(NEW_PATH, v + '_gt')):
        gt_path = os.path.join(NEW_PATH, v + '_gt', gt)
        img = cv2.imread(os.path.join(NEW_PATH, v, gt.split('.')[0] + '.jpg'))
        with open(gt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                obj_id, x, y, w, h, track_id = line[:]
                x, y, w, h = int(x), int(y), int(w), int(h)
                img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
                img = cv2.putText(img,  cls[int(obj_id)] + track_id, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(NEW_PATH, v, gt.split('.')[0] + '.jpg'), img)
            
