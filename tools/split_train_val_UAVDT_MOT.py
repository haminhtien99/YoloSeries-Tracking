import os
import shutil
import random

source_path = '/home/ha/Downloads/Dataset/UAVDT-MOT'
gts = os.path.join(source_path, 'gt')
gt_whole = os.path.join(source_path, 'gt_whole')
train_path = os.path.join(source_path, 'train')
if not os.path.exists(train_path):
    os.makedirs(train_path)
val_path = os.path.join(source_path, 'val')
if not os.path.exists(val_path):
    os.makedirs(val_path)


train_video_path = os.path.join(train_path,'sequences')
if not os.path.exists(train_video_path):
    os.makedirs(train_video_path)
val_video_path = os.path.join(val_path,'sequences')
if not os.path.exists(val_video_path):
    os.makedirs(val_video_path)


train_gt_path = os.path.join(train_path, 'gt')
if not os.path.exists(train_gt_path):
    os.makedirs(train_gt_path)
val_gt_path = os.path.join(val_path, 'gt')
if not os.path.exists(val_gt_path):
    os.makedirs(val_gt_path)


videos = os.listdir(source_path)
for v in videos:
    if v == 'gt' or v == 'train' or v == 'val':
        videos.remove(v)

random.shuffle(videos)
train_videos = videos[:40]
val_videos = videos[40:]
for v in train_videos:
    video_path = os.path.join(source_path, v)
    shutil.move(video_path, train_video_path)
    shutil.move(os.path.join(gts, v + '_gt.txt'), train_gt_path)


for v in val_videos:
    video_path = os.path.join(source_path, v)
    shutil.move(video_path, val_video_path)
    shutil.move(os.path.join(gts, v + '_gt.txt'), val_gt_path)

for split in ['train', 'val']:
    videos = os.listdir(os.path.join(source_path, split,'sequences'))
    path = os.path.join(source_path, split, 'gt_whole')
    if not os.path.exists(path):
        os.makedirs(path)
    for v in videos:
        shutil.move(os.path.join(source_path, gt_whole, v + '_gt_whole.txt'), path)
