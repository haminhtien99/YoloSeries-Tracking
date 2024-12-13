'''
convert gt.txt from <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<confidence>,<cls_id,<visibality>
               to <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<cls_id>,<confidence>,<x>,<y>,<z>  sorted by frame_index >> target_id (mot15 format)
'''
import os
from tqdm import tqdm
import shutil
MOT_PATH = '/home/ha/Downloads/Dataset/UAVDT-2024-MOT'
for spl in ['train', 'val']:
    spl_set = os.path.join(MOT_PATH, spl)
    for v in tqdm(os.listdir(spl_set), desc=f'UAVDT-2024-{spl}'):
        old_gt_path = os.path.join(spl_set, v, 'gt', 'gt.txt')
        new_gt_folder = os.path.join(spl_set, v, 'gt_mot_challenge')
        if not os.path.exists(new_gt_folder):
            os.makedirs(new_gt_folder)
        else:
            shutil.rmtree(new_gt_folder)
            os.makedirs(new_gt_folder)
        with open(old_gt_path, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            line = line.strip().split(',')
            new_line = [int(i) for i in line[:6]]
            new_lines.append(new_line)
        new_lines.sort(key=lambda x: (x[0], x[1]))
        new_lines = [','.join(map(str, line)) + ',1,-1,-1,-1\n' for line in new_lines]
        with open(os.path.join(new_gt_folder, 'gt.txt'), 'w') as f:
            f.writelines(new_lines)