# Convert annotation bbox to YOLO bbox format
# Use only classes car, truck, bus

import os
from tqdm import tqdm
import cv2
DATA_PATH = '/home/ha/Downloads/Dataset/VisDrone-DET'
SPLITS = ['VisDrone2019-DET-train',
          'VisDrone2019-DET-val',
          'VisDrone2019-DET-test-dev']
id2cls_visdrone = {0: 'pedestrian',
                   1: 'people',
                   2: 'bicycle',
                   3: 'car',
                   4: 'van',
                   5: 'truck',
                   6: 'tricycle',
                   7: 'awning-tricycle',
                   8: 'bus',
                   9: 'motor'}

for split in SPLITS:
    data_path = os.path.join(DATA_PATH, split)
    output_path = os.path.join(DATA_PATH, split, 'labels')
    img_dir = os.path.join(data_path, 'images')
    imgs = sorted(os.listdir(img_dir))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for img_name in tqdm(imgs, desc=f'{split}'):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        ann_file = img_name.split('.')[0] + '.txt'
        ann_path = os.path.join(data_path, 'annotations', ann_file)
        yolo_path = os.path.join(output_path, ann_file)
        with open(ann_path, 'r', encoding='utf8') as f:
            for line in f:
                data = line.strip().split(',')
                id_visdrone = int(data[5]) - 1
                considered = int(data[4])
                if id_visdrone >= 0 and id_visdrone <= 9 and considered != 0:
                    cls = id2cls_visdrone[id_visdrone]
                    # create new dataset with only car, truck and bus
                    if cls == 'car':
                        id = 0
                    elif cls == 'truck':
                        id = 1
                    elif cls == 'bus':
                        id = 2
                    else:
                        id = -1
                    if id != -1:
                        x1, y1, w, h = map(float, data[:4])
                        x_center = (x1 + w/2) / width
                        y_center = (y1 + h/2) / height
                        w /= width
                        h /= height
                        with open(yolo_path, 'a+', encoding='utf8') as output_file:
                            output_file.write(f'{id} {x_center} {y_center} {w} {h}\n')
