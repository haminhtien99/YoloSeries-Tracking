import os
import cv2
from tqdm import tqdm
import shutil

visdrone_path = '/home/ha/Downloads/Dataset/VisDrone2019-vehicles-MOT'
uavdt_path = '/home/ha/Downloads/Dataset/UAVDT-2024-MOT'
out_path = '/home/ha/Downloads/Dataset/CustomVehicle-ReID'
def crop(path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    init_id = len(os.listdir(out_path)) + 1
    splits = os.listdir(path)
    for spl in splits:
        if spl.endswith('.md'):
            continue
        videos = os.listdir(os.path.join(path, spl))
        videos.sort()
        for v in videos:
            global_ids = {}
            imgs_path = os.path.join(path, spl, v, 'img1')
            imgs = os.listdir(imgs_path)
            imgs.sort()
            gt_path = os.path.join(path, spl, v, 'gt_mot_challenge', 'gt.txt')
            with open(gt_path, 'r') as f:
                lines = f.readlines()
            new_lines = {}
            for line in lines:
                line = line.strip().split(',')
                frame_id = int(line[0]) - 1
                if frame_id % 20 != 0:
                    continue
                img_name = imgs[frame_id]
                if img_name not in new_lines:
                    new_lines[img_name] = {}
                track_id = int(line[1])
                if track_id not in global_ids:
                    global_ids[track_id] = init_id
                    init_id += 1
                bbox = [int(i) for i in line[2:6]]
                new_lines[img_name][track_id] = bbox
            for name in tqdm(new_lines.keys(), desc=f'{path}/{spl}/{v}'):
                img = cv2.imread(os.path.join(imgs_path, name))
                for track_id, bbox in new_lines[name].items():
                    global_id = global_ids[track_id]
                    new_path = os.path.join(out_path, f'{global_id:05d}_{spl}_{v}')
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    x, y, w, h = bbox
                    if w < 32 or h < 32:
                        continue
                    crop_image = img[y:y+h, x:x+w]
                    cv2.imwrite(os.path.join(new_path, f'{global_id:05d}_{spl}_{v}_{name}'), crop_image)


def make_pytorch(outpath):
    count = 0
    pytorch_path = os.path.join(out_path, 'pytorch')
    if os.path.exists(pytorch_path):
        shutil.rmtree(pytorch_path)
    ids = os.listdir(out_path)
    os.makedirs(pytorch_path)
    train_path = os.path.join(pytorch_path, 'train')
    query_path = os.path.join(pytorch_path, 'query')
    gallery_path = os.path.join(pytorch_path, 'gallery')
    os.makedirs(train_path)
    os.makedirs(query_path)
    
    for id in ids:
        old_path_id = os.path.join(out_path, id)
        if len(os.listdir(old_path_id)) < 5:
            shutil.rmtree(old_path_id)
            continue
        count += 1
        global_id = id.split('_')[0]
        if 'train' in id:
            new_path = os.path.join(train_path, global_id)
        else:   # query
            new_path = os.path.join(query_path, global_id)

        shutil.copytree(old_path_id, new_path)
        shutil.rmtree(old_path_id)
    os.makedirs(gallery_path)
    for id in os.listdir(query_path):
        querys = os.listdir(os.path.join(query_path, id))
        first_img = querys[0]
        os.makedirs(os.path.join(gallery_path, id))
        shutil.move(os.path.join(query_path, id, first_img), os.path.join(gallery_path, id, first_img))
    print(count)

crop(uavdt_path, out_path)
crop(visdrone_path, out_path)
make_pytorch(out_path)


# manually delete bad object ID
# rewrite id object in final set

def check_query_gallery(pytorch_path: str):
    query_path = os.path.join(pytorch_path, 'query')
    gallery_path = os.path.join(pytorch_path, 'gallery')
    querys = os.listdir(query_path)
    gallerys = os.listdir(gallery_path)
    for q in querys:
        if q not in gallerys:
            return False
    return True

def rewrite_pytorch(pytorch_path: str):
    if not check_query_gallery(pytorch_path):
        print('failed to create test set')
        return
    train_path = os.path.join(pytorch_path, 'train')
    query_path = os.path.join(pytorch_path, 'query')
    gallery_path = os.path.join(pytorch_path, 'gallery')
    id_train_list = os.listdir(train_path)
    id_train_list.sort()
    for i, id in tqdm(enumerate(id_train_list), desc='train'):
        new_id = i + 1
        os.makedirs(os.path.join(train_path, f'{new_id:04d}'))
        old_list = os.listdir(os.path.join(train_path, id))
        for name in old_list:
            old_path = os.path.join(train_path, id, name)
            name = name.split('_')
            name[0] = f'{new_id:04d}'
            new_name = '_'.join(name)
            new_path = os.path.join(train_path, f'{new_id:04d}', new_name)
            shutil.copy(old_path, new_path)
        shutil.rmtree(os.path.join(train_path, id))
    id_query_list = os.listdir(query_path)
    id_query_list.sort()
    for i, id in tqdm(enumerate(id_query_list), desc='query'):
        new_id = i + len(id_train_list) + 1
        os.makedirs(os.path.join(query_path, f'{new_id:04d}'))
        old_list = os.listdir(os.path.join(query_path, id))
        for name in old_list:
            old_path = os.path.join(query_path, id, name)
            name = name.split('_')
            name[0] = f'{new_id:04d}'
            new_name = '_'.join(name)
            new_path = os.path.join(query_path, f'{new_id:04d}', new_name)
            shutil.copy(old_path, new_path)

        os.makedirs(os.path.join(gallery_path, f'{new_id:04d}'))
        old_list = os.listdir(os.path.join(gallery_path, id))
        for name in old_list:
            old_path = os.path.join(gallery_path, id, name)
            name = name.split('_')
            name[0] = f'{new_id:04d}'
            new_name = '_'.join(name)
            new_path = os.path.join(gallery_path, f'{new_id:04d}', new_name)
            shutil.copy(old_path, new_path)
        shutil.rmtree(os.path.join(query_path, id))
        shutil.rmtree(os.path.join(gallery_path, id))
pytorch_path = os.path.join(out_path, 'pytorch')
rewrite_pytorch(pytorch_path)