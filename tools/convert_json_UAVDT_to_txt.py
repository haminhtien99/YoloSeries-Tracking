import json
from pycocotools import coco
import os
import cv2
# Specify the path to your JSON file
folder_path = '/home/ha/Downloads/Dataset/uavdt-DatasetNinja-remake/train'
annotations_json = folder_path + '/ann/'
annotations_txt = folder_path + '/annotations/'  # pascal_voc annotations 'cls_Id, x_min, y_min, x_max, y_max'
labels = folder_path + '/labels/'   # YOLO labels 'cls_Id, x_center, y_center, width, height'
if not os.path.exists(annotations_txt):
    os.makedirs(annotations_txt)
if not os.path.exists(labels):
    os.makedirs(labels)
for ann in os.listdir(annotations_json):
    name = '.'.join(ann.split('.')[:-2])
    ann_txt = annotations_txt + name + '.txt'
    with open(annotations_json + ann, 'r') as file:
        data = json.load(file)
        objects = data['objects']
        size = data['size']
    with open(ann_txt, 'w') as file:
        for obj in objects:
            cls_Id = obj['classId']
            if cls_Id == 6544594:  # car
                cls_Id = '0'
            if cls_Id == 6544595:  # truck
                cls_Id = '1'
            if cls_Id == 6544596:  # bus
                cls_Id = '2'
            if cls_Id == 6544597:  # vehicle
                cls_Id = '3'
            pts = obj['points']['exterior']
            x1 = str(pts[0][0])
            y1 = str(pts[0][1])
            x2 = str(pts[1][0])
            y2 = str(pts[1][1])
            string = ','.join([cls_Id, x1, y1, x2, y2])
            file.write(string + '\n')
    label = labels + name + '.txt'
    height = size['height']
    width = size['width']
    with open(label, 'w') as file:
        for obj in objects:
            cls_Id = obj['classId']
            if cls_Id == 6544594:  # car
                cls_Id = '0'
            if cls_Id == 6544595:  # truck
                cls_Id = '1'
            if cls_Id == 6544596:  # bus
                cls_Id = '2'
            if cls_Id == 6544597:  # vehicle, not used
                cls_Id = '3'
            pts = obj['points']['exterior']
            x_center = (float(pts[0][0]) + float(pts[1][0])) / 2
            y_center = (float(pts[0][1]) + float(pts[1][1])) / 2 
            x_center = str(x_center / width)
            y_center = str(y_center/ height)
            w = str((float(pts[1][0]) - float(pts[0][0])) / width)
            h = str((float(pts[1][1]) - float(pts[0][1])) / height)
            string = ' '.join([cls_Id, x_center, y_center, w, h])
            file.write(string + '\n')
print(len(os.listdir(annotations_json)), 'Done!')