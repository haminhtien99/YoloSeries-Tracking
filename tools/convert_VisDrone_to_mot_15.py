'''
convert annotations *.txt 

from <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

to <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<cls_id>,<confidence>,<x>,<y>,<z>

sorted by frame_index >> target_id (mot15 format)
'''
import os
from tqdm import tqdm
import shutil
import configparser
import cv2
MOT_PATH = '/home/ha/Downloads/Dataset/VisDrone2019-MOT'
def convert_visdrone_to_mot15():
    for spl in ['VisDrone2019-MOT-train', 'VisDrone2019-MOT-val']:
        spl_set = os.path.join(MOT_PATH, spl)
        annotations_folder = os.path.join(spl_set, 'annotations')
        gt_folder = os.path.join(spl_set, 'gt_mot_challenge')
        if not os.path.exists(gt_folder):
            os.makedirs(gt_folder)
        else:
            shutil.rmtree(gt_folder)
            os.makedirs(gt_folder)
        for ann_video in tqdm(os.listdir(annotations_folder), desc=f'{spl}'):
            video = ann_video.split('.')[0]
            with open(os.path.join(annotations_folder, ann_video), 'r') as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                line = line.strip().split(',')
                new_line = [int(i) for i in line[:6]]
                if int(line[7]) > 10 and int(line[7]) < 1: # ignore_region : 0, others : 11
                    continue
                new_lines.append(new_line)
            new_lines.sort(key=lambda x: (x[0], x[1]))
            new_lines = [','.join(map(str, line)) + ',1,-1,-1,-1\n' for line in new_lines]
            if not os.path.exists(os.path.join(gt_folder, video)):
                os.makedirs(os.path.join(gt_folder, video))
            gt_file = os.path.join(gt_folder, video, 'gt.txt')
            with open(gt_file, 'w') as f:
                f.writelines(new_lines)
# convert_visdrone_to_mot15()
# Remove videos and annotations that do not contain cars, trucks, buses and vans
NEW_MOT_PATH = '/home/ha/Downloads/Dataset/VisDrone2019-vehicles-MOT'
def remove_unused():
    id2cls_visdrone = {0: 'ignore',
                       1: 'pedestrian',
                       2: 'people',
                       3: 'bicycle',
                       4: 'car',
                       5: 'van',
                       6: 'truck',
                       7: 'tricycle',
                       8: 'awning-tricycle',
                       9: 'bus',
                       10: 'motor',
                       11: 'others'}
    new_cls_2_id = {'car': 0, 'truck': 1, 'bus': 2, 'van': 3}

    for spl in ['VisDrone2019-MOT-train', 'VisDrone2019-MOT-val']:
        spl_set = os.path.join(MOT_PATH, spl)
        new_spl_set = os.path.join(NEW_MOT_PATH, spl)
        annotations_folder = os.path.join(spl_set, 'annotations')
        for ann in tqdm(os.listdir(annotations_folder), desc=f'new mot dataset{spl}'):
            video_name = ann.split('.')[0]
            with open(os.path.join(annotations_folder, ann), 'r') as f:
                lines = f.readlines()
                new_lines = []
                for line in lines:
                    line = line.strip().split(',')
                    new_line = [int(i) for i in line]
                    cls_id = new_line[7]
                    cls_name = id2cls_visdrone[cls_id]
                    if cls_name not in new_cls_2_id.keys():
                        continue
                    new_line[7] = new_cls_2_id[cls_name]
                    new_lines.append(new_line)                    
            # copy only videos that contain cars, trucks, buses and vans
            if len(new_lines) == 0:
                continue

            new_annotations_folder = os.path.join(new_spl_set, video_name, 'annotations')
            new_file = os.path.join(new_annotations_folder, ann)
            new_lines_to_annotation = [','.join(map(str, line)) + '\n' for line in new_lines]
            if not os.path.exists(new_annotations_folder):
                os.makedirs(new_annotations_folder)
            with open(new_file, 'w') as f:
                f.writelines(new_lines_to_annotation)
            
            new_lines_to_gt = [line[:6] for line in new_lines]
            new_lines_to_gt.sort(key=lambda x: (x[0], x[1]))
            new_lines_to_gt = [','.join(map(str, line)) + ',1,-1,-1,-1\n' for line in new_lines_to_gt]
            new_gt_folder = os.path.join(new_spl_set, video_name, 'gt_mot_challenge')
            if not os.path.exists(new_gt_folder):
                os.makedirs(new_gt_folder)
            with open(os.path.join(new_gt_folder, 'gt.txt'), 'w') as f:
                f.writelines(new_lines_to_gt)

            sequence_folder = os.path.join(spl_set, 'sequences', video_name)
            new_images_folder = os.path.join(new_spl_set, video_name, 'img1')
            if not os.path.exists(new_images_folder):
                os.makedirs(new_images_folder)
            for img_name in os.listdir(sequence_folder):
                shutil.copy(os.path.join(sequence_folder, img_name), os.path.join(new_images_folder, img_name))

            sample_img = cv2.imread(os.path.join(sequence_folder, img_name))
            frame_width, frame_height = sample_img.shape[:2]
            config = configparser.ConfigParser()
            config['Sequence'] = {
                'name': video_name,
                'imDir': 'img1',
                'frameRate': '30',
                'seqLength': str(len(os.listdir(sequence_folder))),
                'imWidth': str(frame_width),
                'imHeight': str(frame_height),
                'imExt': '.jpg'
            }
            with open(os.path.join(new_spl_set, video_name, 'seqinfo.ini'), 'w') as f:
                config.write(f)
        print(len(os.listdir(new_spl_set)))
# convert_visdrone_to_mot15()
remove_unused()
        
