import os
from tqdm import tqdm
import shutil
import configparser
import cv2
GT_PATH = '/home/ha/Downloads/Dataset/UAV-benchmark-MOTD_v1.0/GT'
NEW_PATH = '/home/ha/Downloads/Dataset/UAVDT-2024'
NEW_GT_PATH = os.path.join(NEW_PATH, 'gt')
if not os.path.exists(NEW_GT_PATH):
    os.makedirs(NEW_GT_PATH)
change_videos = ['M0901', 'M1005', 'M1006', 'M1008', 'M1009', 'M1102', 'M1202']
# M1202
def M1202():
    gt_file = 'M1202_gt_whole.txt'
    gt_path = os.path.join(GT_PATH, gt_file)
    with open(gt_path, 'r') as f:
        lines = f.readlines()
        new_lines = []
        for line in tqdm(lines, desc='M1202'):
            line = line.strip().split(',')
            target_id = line[1]
            if target_id == '11' or target_id == '10':
                line[-1] = '3' # van
            elif line[-1] == '1':
                line[-1] = '0' # car
            elif line[-1] == '2':
                line[-1] = '1' # truck
            elif line[-1] == '3':
                line[-1] = '2' # bus
            line = ','.join(line)
            line += '\n'
            new_lines.append(line)
    old_video_path = '/home/ha/Downloads/Dataset/UAV-benchmark-M/M1202'
    images = os.listdir(old_video_path)
    images.sort()
    new_video_path = os.path.join(NEW_PATH, 'M1202')
    if not os.path.exists(new_video_path):
        os.makedirs(new_video_path)
    new_gt_path = os.path.join(NEW_GT_PATH, gt_file)
    with open(new_gt_path, 'w') as f:
        f.writelines(new_lines)
    groups = {}
    for line in new_lines:
        line = line.strip().split(',')
        frame_id = line[0]
        if frame_id not in groups:
            groups[frame_id] = []
        x, y, w, h = line[2:6]
        x, y, w, h = int(x), int(y), int(w), int(h)
        score = 1
        cls_id = int(line[-1])
        truncation = 0
        occulation = 0
        groups[frame_id].append(f'{x},{y},{w},{h},{score},{cls_id},{truncation},{occulation}\n')
    for frame_id, objects in groups.items():
        frame_id = int(frame_id)
        img_name = images[frame_id-1]
        annotation_folder = os.path.join(new_video_path, 'annotations')
        if not os.path.exists(annotation_folder):
            os.makedirs(annotation_folder)
        ann_txt = os.path.join(annotation_folder ,img_name.split('.')[0] + '.txt')
        with open(ann_txt, 'w') as f:
            f.writelines(objects)       
def M0901():
    gt_file = 'M0901_gt_whole.txt'
    gt_path = os.path.join(GT_PATH, gt_file)
    old_video_path = '/home/ha/Downloads/Dataset/UAV-benchmark-M/M0901'
    images = os.listdir(old_video_path)
    images.sort()
    new_video_path = os.path.join(NEW_PATH, 'M0901')
    if not os.path.exists(new_video_path):
        os.makedirs(new_video_path)
    with open(gt_path, 'r') as f:
        lines = f.readlines()
    groups = {}
    for line in lines:
        line = line.strip().split(',')
        frame_id = line[0]
        if frame_id not in groups:
            groups[frame_id] = []
        x, y, w, h = line[2:6]
        x, y, w, h = int(x), int(y), int(w), int(h)
        score = 1
        cls_id = int(line[-1]) - 1
        truncation = 0
        occulation = 0
        groups[frame_id].append(f'{x},{y},{w},{h},{score},{cls_id},{truncation},{occulation}\n')
    for frame_id, objects in groups.items():
        frame_id = int(frame_id)
        img_name = images[frame_id-1]
        annotation_folder = os.path.join(new_video_path, 'annotations')
        if not os.path.exists(annotation_folder):
            os.makedirs(annotation_folder)
        ann_txt = os.path.join(annotation_folder ,img_name.split('.')[0] + '.txt')
        with open(ann_txt, 'w') as f:
            f.writelines(objects)

def M1005():
    gt_file = 'M1005_gt_whole.txt'
    gt_path = os.path.join(GT_PATH, gt_file)
    with open(gt_path, 'r') as f:
        lines = f.readlines()
        new_lines = []
        for line in tqdm(lines, desc='M1005'):
            line = line.strip().split(',')
            target_id = line[1]
            if target_id == '4' or target_id == '8':
                line[-1] = '3' # van
            elif line[-1] == '1':
                line[-1] = '0' # car
            elif line[-1] == '2':
                line[-1] = '1' # truck
            elif line[-1] == '3':
                line[-1] = '2' # bus
            line = ','.join(line)
            line += '\n'
            new_lines.append(line)
    old_video_path = '/home/ha/Downloads/Dataset/UAV-benchmark-M/M1005'
    images = os.listdir(old_video_path)
    images.sort()
    new_video_path = os.path.join(NEW_PATH, 'M1005')
    if not os.path.exists(new_video_path):
        os.makedirs(new_video_path)
    new_gt_path = os.path.join(NEW_GT_PATH, gt_file)
    with open(new_gt_path, 'w') as f:
        f.writelines(new_lines)
    groups = {}
    for line in new_lines:
        line = line.strip().split(',')
        frame_id = line[0]
        if frame_id not in groups:
            groups[frame_id] = []
        x, y, w, h = line[2:6]
        x, y, w, h = int(x), int(y), int(w), int(h)
        score = 1
        cls_id = int(line[-1])
        truncation = 0
        occulation = 0
        groups[frame_id].append(f'{x},{y},{w},{h},{score},{cls_id},{truncation},{occulation}\n')
    for frame_id, objects in groups.items():
        frame_id = int(frame_id)
        img_name = images[frame_id-1]
        annotation_folder = os.path.join(new_video_path, 'annotations')
        if not os.path.exists(annotation_folder):
            os.makedirs(annotation_folder)
        ann_txt = os.path.join(annotation_folder ,img_name.split('.')[0] + '.txt')
        with open(ann_txt, 'w') as f:
            f.writelines(objects)
def M1006():
    gt_file = 'M1006_gt_whole.txt'
    gt_path = os.path.join(GT_PATH, gt_file)
    with open(gt_path, 'r') as f:
        lines = f.readlines()
        new_lines = []
        for line in tqdm(lines, desc='M1006'):
            line = line.strip().split(',')
            target_id = line[1]
            if target_id == '1' or target_id == '2' or target_id == '27' or target_id == '22':
                line[-1] = '3' # van
            elif line[-1] == '1':
                line[-1] = '0' # car
            elif line[-1] == '2':
                line[-1] = '1' # truck
            elif line[-1] == '3':
                line[-1] = '2' # bus
            line = ','.join(line)
            line += '\n'
            new_lines.append(line)
    
    old_video_path = f'/home/ha/Downloads/Dataset/UAV-benchmark-M/M1006'
    images = os.listdir(old_video_path)
    images.sort()
    new_video_path = os.path.join(NEW_PATH, 'M1006')
    if not os.path.exists(new_video_path):
        os.makedirs(new_video_path)
    new_gt_path = os.path.join(NEW_GT_PATH, gt_file)
    with open(new_gt_path, 'w') as f:
        f.writelines(new_lines)
    groups = {}
    for line in new_lines:
        line = line.strip().split(',')
        frame_id = line[0]
        if frame_id not in groups:
            groups[frame_id] = []
        x, y, w, h = line[2:6]
        x, y, w, h = int(x), int(y), int(w), int(h)
        score = 1
        cls_id = int(line[-1])
        truncation = 0
        occulation = 0
        groups[frame_id].append(f'{x},{y},{w},{h},{score},{cls_id},{truncation},{occulation}\n')
    for frame_id, objects in groups.items():
        frame_id = int(frame_id)
        img_name = images[frame_id-1]
        annotation_folder = os.path.join(new_video_path, 'annotations')
        if not os.path.exists(annotation_folder):
            os.makedirs(annotation_folder)
        ann_txt = os.path.join(annotation_folder ,img_name.split('.')[0] + '.txt')
        with open(ann_txt, 'w') as f:
            f.writelines(objects)
def M1009():
    gt_file = 'M1009_gt_whole.txt'
    gt_path = os.path.join(GT_PATH, gt_file)
    with open(gt_path, 'r') as f:
        lines = f.readlines()
        new_lines = []
        for line in tqdm(lines, desc='M1009'):
            line = line.strip().split(',')
            target_id = line[1]
            if target_id == '9':
                line[-1] = '2' # truck --> bus
            elif target_id == '22':
                line[-1] = '3' # van
            elif line[-1] == '1':
                line[-1] = '0' # car
            elif line[-1] == '2':
                line[-1] = '1' # truck
            elif line[-1] == '3':
                line[-1] = '2' # bus
            line = ','.join(line)
            line += '\n'
            new_lines.append(line)

    old_video_path = f'/home/ha/Downloads/Dataset/UAV-benchmark-M/M1009'
    images = os.listdir(old_video_path)
    images.sort()
    new_video_path = os.path.join(NEW_PATH, 'M1009')
    if not os.path.exists(new_video_path):
        os.makedirs(new_video_path)
    new_gt_path = os.path.join(NEW_GT_PATH, gt_file)
    with open(new_gt_path, 'w') as f:
        f.writelines(new_lines)
    groups = {}
    for line in new_lines:
        line = line.strip().split(',')
        frame_id = line[0]
        if frame_id not in groups:
            groups[frame_id] = []
        x, y, w, h = line[2:6]
        x, y, w, h = int(x), int(y), int(w), int(h)
        score = 1
        cls_id = int(line[-1])
        truncation = 0
        occulation = 0
        groups[frame_id].append(f'{x},{y},{w},{h},{score},{cls_id},{truncation},{occulation}\n')
    for frame_id, objects in groups.items():
        frame_id = int(frame_id)
        img_name = images[frame_id-1]
        annotation_folder = os.path.join(new_video_path, 'annotations')
        if not os.path.exists(annotation_folder):
            os.makedirs(annotation_folder)
        ann_txt = os.path.join(annotation_folder ,img_name.split('.')[0] + '.txt')
        with open(ann_txt, 'w') as f:
            f.writelines(objects)
def rewrite_gt_UAVDT():
    # change some cars to vans
    changes = {'M1006': [1, 2, 27, 22],
               'M1008': [], # nothing change
               'M0901': [], # nothing change
               'M1005': [4, 8],
               'M1009': [22],
               'M1202': [11, 10],
               'M1102': [54, 32, 16, 1, 4]}
    for video_name, target_ids in changes.items():
        gt_file = f'{video_name}_gt_whole.txt'
        gt_path = os.path.join(GT_PATH, gt_file)
        with open(gt_path, 'r') as f:
            lines = f.readlines()
            new_lines = []
            for line in tqdm(lines, desc=f'{video_name}'):
                line = line.strip().split(',')
                frame_id = int(line[0])
                target_id = int(line[1])
                obj_id = int(line[-1]) - 1
                x, y, w, h = line[2:6]
                x, y, w, h = int(x), int(y), int(w), int(h)
                if target_id in target_ids:
                    obj_id = 3 # van
                if video_name == 'M1009' and target_id == 9:
                    obj_id = 2 # truck --> bus
                new_line = [frame_id, target_id, x, y, w, h, 1, obj_id, 1]
                new_line = ','.join(map(str, new_line)) + '\n'
                new_lines.append(new_line)
        old_video_path = f'/home/ha/Downloads/Dataset/UAV-benchmark-M/{video_name}'
        images = os.listdir(old_video_path)
        images.sort()
        new_video_path = os.path.join(NEW_PATH, video_name)
        if not os.path.exists(new_video_path):
            os.makedirs(new_video_path)
        new_gt_path = os.path.join(NEW_GT_PATH, f'{video_name}_gt.txt')
        with open(new_gt_path, 'w') as f:
            f.writelines(new_lines)
# rewrite_gt_UAVDT()
MOT_PATH = '/home/ha/Downloads/Dataset/UAVDT-2024-MOT'
def copy_gt_UAVDT_from_MOT():
    videos = os.listdir(NEW_PATH)
    videos = [v for v in videos if v.startswith('M') and v not in change_videos]
    videos.sort()
    for v in videos:
        gt_path = os.path.join(MOT_PATH, v, 'gt', 'gt.txt')
        new_gt_path = os.path.join(NEW_GT_PATH, f'{v}_gt.txt')
        shutil.copy(gt_path, new_gt_path)
def copy_gt_MOT_from_UAVDT():
    for v in change_videos:
        gt_folder = os.path.join(MOT_PATH, v, 'gt')
        if not os.path.exists(gt_folder):
            os.makedirs(gt_folder)
        gt_path = os.path.join(NEW_GT_PATH, f'{v}_gt.txt')
        new_gt_path = os.path.join(gt_folder, 'gt.txt')
        shutil.copy(gt_path, new_gt_path)

        # write seqinfo.ini
        seqLength = len(os.listdir(os.path.join(MOT_PATH, v, 'img1')))
        sample_img = os.path.join(MOT_PATH, v, 'img1', '000001.jpg')
        img = cv2.imread(sample_img)
        frame_width, frame_height = img.shape[:2]
        config = configparser.ConfigParser()
        config['Sequence'] = {
            'name': v,
            'imDir': 'img1',
            'frameRate': '30',
            'seqLength': str(seqLength),
            'imWidth': str(frame_width),
            'imHeight': str(frame_height)
        }
        with open(os.path.join(MOT_PATH, v, 'seqinfo.ini'), 'w') as f:
            config.write(f)
def rewrite_gt_MOT_challenge():
    # frame_id, target_id, x, y, w, h, confidence, x_3D, y_3D, z_3D - 3D coordinates
    for spl in ['test', 'train']:
        spl_path = os.path.join(MOT_PATH, spl)
        videos = os.listdir(spl_path)
        for v in videos:
            gt_path = os.path.join(spl_path, v, 'gt', 'gt.txt')
            new_gt_folder = os.path.join(spl_path, v, 'gt_mot_challenge')
            if not os.path.exists(new_gt_folder):
                os.makedirs(new_gt_folder)
            new_gt_path = os.path.join(new_gt_folder, 'gt.txt')
            with open(gt_path, 'r') as f:
                lines = f.readlines()
                new_lines = []
                for line in lines:
                    line = line.strip().split(',')
                    new_line = [line[0], line[1], line[2], line[3], line[4], line[5], '-1', '-1', '-1', '-1']
                    new_line = ','.join(new_line) + '\n'
                    new_lines.append(new_line)
            with open(new_gt_path, 'w') as f:
                    f.writelines(new_lines)
rewrite_gt_MOT_challenge()