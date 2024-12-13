import os
import shutil
import cv2
from tqdm import tqdm

DATAPATH = '/home/ha/Downloads/Dataset/UAVDT-2024'
val_set = ['M0205','M0208','M0403','M0601',
           'M0606','M0802','M1001',
           'M1009','M1101','M1301','M1302']
val_set = [i for i in val_set if i in os.listdir(DATAPATH)]
train_set = [i for i in os.listdir(DATAPATH) if i not in val_set and i.startswith('M')]
spl_set = {'train': train_set, 'val': val_set}

DET_PATH = '/home/ha/Downloads/Dataset/UAVDT-2024-DET'
cls2obj = {'0': 'car', '1': 'truck', '2': 'bus', '3': 'van'}
if not os.path.exists(DET_PATH):
    os.mkdir(DET_PATH)
for spl, set in spl_set.items():
    spl_path = os.path.join(DET_PATH, spl)
    if not os.path.exists(spl_path):
        os.mkdir(spl_path)
    else:
        shutil.rmtree(spl_path)
        os.mkdir(spl_path)
    images_path = os.path.join(DET_PATH, spl, 'images')
    os.mkdir(images_path)

    annotations_path = os.path.join(DET_PATH, spl, 'annotations')
    os.mkdir(annotations_path)
    
    labels_path = os.path.join(DET_PATH, spl, 'labels')
    os.mkdir(labels_path)

    images_with_bbox = os.path.join(DET_PATH, spl, 'images_with_bbox')
    os.mkdir(images_with_bbox)

    for v in tqdm(set, desc=spl):
        images_in_video = os.listdir(os.path.join(DATAPATH, v, 'images'))
        images_in_video.sort()
        for i, img_name in enumerate(images_in_video):
            if i % 10 != 0:
                continue
            src_img_path = os.path.join(DATAPATH, v, 'images', img_name)
            new_img_name = v + '_' + img_name
            dst_img_path = os.path.join(images_path, new_img_name)
            shutil.copy2(src_img_path, dst_img_path)

            ann_name = img_name.split('.')[0] + '.txt'
            src_ann_path = os.path.join(DATAPATH, v, 'annotations', ann_name)
            new_ann_name = v + '_' + ann_name
            dst_ann_path = os.path.join(annotations_path, new_ann_name)
            shutil.copy2(src_ann_path, dst_ann_path)
            
            img = cv2.imread(dst_img_path)
            draw_img = img.copy()
            height, width, _ = img.shape
            with open(dst_ann_path, 'r') as f:
                lines = f.readlines()
                new_lines = []
                for line in lines:
                    line = line.strip().split(',')
                    x, y, w, h = line[:4]
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    draw_img = cv2.rectangle(draw_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    x_center = (x + w/2)/width
                    y_center = (y + h/2)/height
                    w = w/width
                    h = h/height
                    cls = line[5]
                    obj = cls2obj[cls]
                    draw_img = cv2.putText(draw_img, str(obj), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.imwrite(os.path.join(images_with_bbox, new_img_name), draw_img)
                    new_line = f'{cls} {x_center} {y_center} {w} {h}\n'
                    new_lines.append(new_line)
            label_path = os.path.join(labels_path, new_ann_name)
            with open(label_path, 'w') as f:
                f.writelines(new_lines)
            