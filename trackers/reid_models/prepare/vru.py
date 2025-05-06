import os
import shutil
from tqdm import tqdm


def read_path(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    res_dict = {}
    number = 0
    for line in lines:
        line = line.strip().split()
        frame_id, obj_id = line
        obj_id = '{:0>5}'.format(obj_id)    # maximum 15,085 vehicles
        if obj_id not in res_dict.keys():
            res_dict[obj_id] = [frame_id]
        else:
            res_dict[obj_id].append(frame_id)
        number += 1
    return res_dict, number

def copy_train_files(train_dict, new_path, pic_path):
    i=0
    sub_new_path = os.path.join(new_path, 'train')
    for obj_id, frames in tqdm(train_dict.items(), desc=pic_path):
        obj_path = os.path.join(sub_new_path, obj_id)
        if not os.path.exists(obj_path):
            os.makedirs(obj_path)
        for frame in frames:
            src_img = os.path.join(pic_path, frame+'.jpg')
            new_img = os.path.join(obj_path, f'{obj_id}_{frame}.jpg')
            shutil.copy(src_img, new_img)
            i += 1
    return i

def copy_test_files(test_dict, new_path, pic_path):
    i=0
    query_path = os.path.join(new_path, 'query')
    gallery_path = os.path.join(new_path, 'gallery')
    os.makedirs(query_path)
    os.makedirs(gallery_path)
    for obj_id, frames in tqdm(test_dict.items(), desc=pic_path):
        obj_query_path = os.path.join(query_path, obj_id)
        os.makedirs(obj_query_path)
        obj_gallery_path = os.path.join(gallery_path, obj_id)
        for frame in frames:
            if not os.path.exists(obj_gallery_path):
                os.makedirs(obj_gallery_path)   # 1 image for gallery
                obj_path = obj_gallery_path
            else:
                obj_path = obj_query_path

            src_img = os.path.join(pic_path, frame+'.jpg')
            new_img = os.path.join(obj_path, f'{obj_id}_{frame}.jpg')
            shutil.copy(src_img, new_img)
            i += 1
    return i


src_path = '/home/ha/Downloads/Dataset/VRU'
pic_path = os.path.join(src_path, 'Pic')
annotations_path = os.path.join(src_path, 'train_test_split')

train_path = os.path.join(annotations_path, 'train.txt')
test_path = os.path.join(annotations_path, 'test_list_1200.txt')    # or test_list_2400.txt, test.txt

train_dict, num_train = read_path(train_path)
test_dict, num_test = read_path(test_path)

new_path = os.path.join(src_path, 'pytorch')
num_train_imgs = copy_train_files(train_dict, new_path, pic_path=pic_path)
num_test_imgs = copy_test_files(test_dict, new_path, pic_path=pic_path)

if num_train_imgs == num_train and num_test_imgs == num_test:
    print('Successfully copied')
    print(f'{num_train_imgs} images in train set and {num_test_imgs} images in test sest')
else:
    print('Failed')
    print(f'{num_train_imgs} vs {num_train}')
    print(f'{num_test_imgs} vs {num_test_imgs}')
