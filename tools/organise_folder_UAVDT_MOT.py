import os
import shutil


source_path = '/home/ha/Downloads/Dataset/uavdt-DatasetNinja-remake' # path to UAVDT-MOT Ninja
train_images_path = os.path.join(source_path, 'train/images')
val_image_path = os.path.join(source_path, 'val/images')

dest_path = '/home/ha/Downloads/Dataset/UAVDT-MOT'
if not os.path.exists(dest_path):
    os.makedirs(dest_path)


for img in os.listdir(train_images_path):
    full_path = os.path.join(train_images_path, img)
    shutil.copy(full_path, dest_path)
for img in os.listdir(val_image_path):
    full_path = os.path.join(val_image_path, img)
    shutil.copy(full_path, dest_path)


path_gt_txt = '/home/ha/Downloads/Dataset/UAV-benchmark-MOTD_v1.0/GT'    # ground truth txt from author UAVDT
images = os.listdir(dest_path)
videos = []
for gt in os.listdir(path_gt_txt):
    if gt.endswith('_gt.txt'):
        videos.append(gt.split('_')[0])
for v in videos:
    count = 0
    dest_path_video = os.path.join(dest_path, v)
    if not os.path.exists(dest_path_video):
        os.makedirs(dest_path_video)
    for img in images:
        if img.endswith('.jpg'):
            if v in img:
                full_path = os.path.join(dest_path, img)
                shutil.move(full_path, dest_path_video)
                count += 1
        
        continue
    print(f'folder {v}: {count}')
dest_path_gt = os.path.join(dest_path, 'gt')
if not os.path.exists(dest_path_gt):
    os.makedirs(dest_path_gt)
for gt in os.listdir(path_gt_txt):
    if gt.endswith('_gt.txt'):
        shutil.copy(os.path.join(path_gt_txt, gt), dest_path_gt)
    continue

dest_path_gt_whole = os.path.join(dest_path, 'gt_whole')
if not os.path.exists(dest_path_gt_whole):
    os.makedirs(dest_path_gt_whole)
for gt in os.listdir(path_gt_txt):
    if gt.endswith('_gt_whole.txt'):
        shutil.copy(os.path.join(path_gt_txt, gt), dest_path_gt_whole)
    continue

