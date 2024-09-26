# convert GT file from UAVDT to YOLO format
import os
from tqdm import tqdm
from glob import glob
import cv2
import json
from typing import Dict, List

cls2id = {0: 'car', 1: 'truck', 2: 'bus'}
data_path = '/home/ha/Downloads/Dataset/UAVDT-MOT'
OUT_PATH = data_path +'/annotations'
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)


image_cnt = 0
ann_cnt = 0
video_cnt = 0
tid_curr = 0

for split in ['train', 'val']:
    out: Dict[str, List] = {'images': [],
                            'annotations': [],
                            'videos': [],
                            'categories': [{'id': k, 'name': v} for k, v in cls2id.items()]
                            }
    split_path = os.path.join(data_path, split)
    sequences = sorted(os.listdir(os.path.join(split_path, 'sequences')))
    for seq in tqdm(sequences, desc=f'{split}: {data_path}'):
        video_cnt += 1  # video sequence number
        out['videos'].append({'id': video_cnt, 'file_name': seq})
        img_dir = os.path.join(split_path,'sequences', seq)
        ann_path = os.path.join(split_path, 'gt_whole', seq+'_gt_whole.txt')
        images = sorted(glob(os.path.join(img_dir, '*.jpg')))
        num_images  = len(images)

    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        image_info = {'file_name': img_path.split('/')[-1],
                      'id': image_cnt + i + 1,
                      'frame_id': i + 1,
                      'prev_image_id': image_cnt + i if i > 0 else -1,
                      'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                      'video_id': video_cnt,
                      'height': height,
                      'width': width}
        out['images'].append(image_info)
    gt_whole = sorted(os.listdir(os.path.join(split_path, 'gt_whole')))
    tid_last = -1
    for gt in gt_whole:
        with open(os.path.join(split_path, 'gt_whole', gt), 'r') as f:
            lines = f.readlines()
            for line in lines:
                ann_cnt += 1
                
                line = line.split(',')
                cat_id = int(line[-1]) - 1        # make cat_id starts from 0

                bbox = [int(i) for i in line[2:6]]
                frame_id = int(line[0])
                track_id = int(line[1])
                if not track_id == tid_last:
                    tid_curr += 1
                    tid_last = track_id

                out_of_view = int(line[6])
                occlusion = int(line[7])
                ann = {'id': ann_cnt,
                       'category_id': cat_id,
                       'image_id': image_cnt + frame_id,
                       'track_id': tid_curr,
                       'bbox': bbox,
                       'conf': 1.0,
                       'iscrowd': 0,
                       'area': float(bbox[2] * bbox[3]),
                       'out_of_view': out_of_view,
                       'occlusion': occlusion}
                out['annotations'].append(ann)
            image_cnt += num_images
    out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
    json.dump(out, open(out_path, 'w'))
        

