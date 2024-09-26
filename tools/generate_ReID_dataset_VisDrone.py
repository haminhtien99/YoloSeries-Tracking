import os
import argparse
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np

def generate_trajectories(file_path):
    f = open(file_path, 'r')
    lines = f.read().split('\n')
    values = []
    for l in lines:
        split = l.split(',')
        if len(split) < 2:
            break
        numbers = [float(i) for i in split]
        values.append(numbers)

    values = np.array(values, np.float_)
    values = np.array(values)
    values[:, 4] += values[:, 2]
    values[:, 5] += values[:, 3]

    return values


split_train_test = ['VisDrone2019-MOT-train', 'VisDrone2019-MOT-val']
data_path = '/home/ha/Downloads/Dataset/VisDrone-MOT'
# Create folder for outputs
save_path = '/home/ha/Downloads/Dataset/Fast-ReID/VisDrone-ReID'
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
train_save_path = os.path.join(save_path, 'bounding_box_train')
os.makedirs(train_save_path, exist_ok=True)
test_save_path = os.path.join(save_path, 'bounding_box_test')
os.makedirs(test_save_path, exist_ok=True)
# Get gt data
for split in split_train_test:
    split_data_path = os.path.join(data_path, split)
    if not os.path.exists(split_data_path):
        os.makedirs(split_data_path, exist_ok=True)
    sequences = sorted(os.listdir(os.path.join(split_data_path, 'sequences')))

    id_offset = 0
    for seq in sequences:
        print(seq)
        print(id_offset)

        ground_truth_path = os.path.join(split_data_path, 'annotations', seq + '.txt')
        gt = generate_trajectories(ground_truth_path)  # f, id, x_tl, y_tl, x_br, y_br, ...

        images_path = os.path.join(split_data_path, 'sequences', seq)
        img_files = os.listdir(images_path)
        img_files.sort()

        num_frames = len(img_files)
        max_id_per_seq = 0
        for f in range(num_frames):

            img = cv2.imread(os.path.join(images_path, img_files[f]))

            if img is None:
                print("ERROR: Receive empty frame")
                continue

            H, W, _ = np.shape(img)

            det = gt[f + 1 == gt[:, 0], 1:].astype(np.int_)

            for d in range(np.size(det, 0)):
                id_ = det[d, 0]
                x1 = det[d, 1]
                y1 = det[d, 2]
                x2 = det[d, 3]
                y2 = det[d, 4]

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x2, W)
                y2 = min(y2, H)

                # patch = cv2.cvtColor(img[y1:y2, x1:x2, :], cv2.COLOR_BGR2RGB)
                patch = img[y1:y2, x1:x2, :]

                max_id_per_seq = max(max_id_per_seq, id_)

                # plt.figure()
                # plt.imshow(patch)
                # plt.show()

                fileName = (str(id_ + id_offset)).zfill(7) + '_' + seq + '_' + (str(f)).zfill(7) + '_acc_data.jpg'

                if f < num_frames // 2:
                    cv2.imwrite(os.path.join(train_save_path, fileName), patch)
                else:
                    cv2.imwrite(os.path.join(test_save_path, fileName), patch)

        id_offset += max_id_per_seq
