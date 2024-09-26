import json
from pycocotools import coco
import os
import shutil
# Specify the path to your JSON file
source_test_path = '/home/ha/Downloads/Dataset/uavdt-DatasetNinja/test/'
source_ann_json = source_test_path + 'ann/'
source_img = source_test_path + 'img/'
dest_val_path = '/home/ha/Downloads/Dataset/uavdt-DatasetNinja/val/'
dest_val_ann_json = dest_val_path + 'ann/'
dest_val_img = dest_val_path + 'images/'
if not os.path.exists(dest_val_path):
    os.makedirs(dest_val_path)
if not os.path.exists(dest_val_ann_json):
    os.makedirs(dest_val_ann_json)
if not os.path.exists(dest_val_img):
    os.makedirs(dest_val_img)

benchmark_M = 0
benchmark_S = 0
for ann in os.listdir(source_ann_json):
    if ann[0] == 'M':
        benchmark_M += 1
        file = source_ann_json + ann
        shutil.copy(file, dest_val_ann_json)
    if ann[0] == 'S':
        benchmark_S += 1
    continue
print(benchmark_M, 'Done!')
print(benchmark_S, 'keep!')

benchmark_M = 0
benchmark_S = 0
for ann in os.listdir(source_img):
    if ann[0] == 'M':
        benchmark_M += 1
        file = source_img + ann
        shutil.copy(file, dest_val_img)
    if ann[0] == 'S':
        benchmark_S += 1
    continue
print(benchmark_M, 'Done!')
print(benchmark_S, 'keep!')