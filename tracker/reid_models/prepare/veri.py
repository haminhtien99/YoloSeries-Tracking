'''
Source from VeRidataset
https://github.com/JDAI-CV/VeRidataset
The dataset is used for non-commercial purposes
Downloaded from another source on kaggle
https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset
'''

import os
from shutil import copyfile
import argparse

def process_data(dir_path: str):
    """Process
    old path:
        dir_path
        |---image_train
        |---image_query
        |---image_test
        ....
    new path:
        dir_path/pytorch
        |---gallery
        |---query
        |---train
        |---train_all
        |---val
    """
    if dir_path is None or not os.path.exists(dir_path):
        raise Exception(f"{dir_path} is not found")
    nums = 0
    nums_dict = {}
    save_path = os.path.join(dir_path, 'pytorch')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #-----------------------------------------
    # query
    query_path = os.path.join(dir_path, 'image_query')
    query_save_path = os.path.join(save_path, 'query')
    if not os.path.exists(query_save_path):
        os.makedirs(query_save_path)
    for _, _, files in os.walk(query_path, topdown=True):
        for name in files:
            if not name.endswith(".jpg"):
                continue
            src_path = os.path.join(query_path, name)
            ID = name.split('_')[0]
            ID_path = os.path.join(query_save_path, ID)
            if not os.path.exists(ID_path):
                os.makedirs(ID_path)
            dst_path = os.path.join(ID_path, name)
            copyfile(src_path, dst_path)
            nums += 1
    nums_dict["query"] = nums
    print(f'query: {nums}')

    #-----------------------------------------
    # train_all
    nums = 0
    train_all_path = os.path.join(dir_path, 'image_train')
    train_all_save_path = os.path.join(save_path, 'train_all')
    if not os.path.exists(train_all_save_path):
        os.makedirs(train_all_save_path)
    for _, _, files in os.walk(train_all_path, topdown=True):
        for name in files:
            if not name.endswith(".jpg"):
                continue
            src_path = os.path.join(train_all_path, name)
            ID = name.split('_')[0]
            ID_path = os.path.join(train_all_save_path, ID)
            if not os.path.exists(ID_path):
                os.makedirs(ID_path)
            dst_path = os.path.join(ID_path, name)
            copyfile(src_path, dst_path)
            nums += 1
    nums_dict["train_all"] = nums
    print(f'train_all: {nums}')

    #-----------------------------------------
    # gallery
    nums = 0
    gallery_path = os.path.join(dir_path, 'image_test')
    query_path = os.path.join(dir_path, 'image_query')
    gallery_save_path = os.path.join(save_path, 'gallery')
    query_files = os.listdir(query_path)
    if not os.path.exists(gallery_save_path):
        os.makedirs(gallery_save_path)
    for _, _, files in os.walk(gallery_path, topdown=True):
        for name in files:
            if not name.endswith(".jpg"):
                continue
            if name in query_files:
                continue # dont use query_files in gallery
            src_path = os.path.join(gallery_path, name)
            ID = name.split('_')[0]
            ID_path = os.path.join(gallery_save_path, ID)
            if not os.path.exists(ID_path):
                os.makedirs(ID_path)
            dst_path = os.path.join(ID_path, name)
            copyfile(src_path, dst_path)
            nums += 1
    nums_dict["gallery"] = nums
    print(f'gallery: {nums}')

    #-----------------------------------------
    # train/val
    nums = 0
    nums_val = 0
    train_path = os.path.join(dir_path, 'image_train')
    train_save_path = os.path.join(save_path, 'train')
    val_save_path = os.path.join(save_path, 'val')
    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)
        os.makedirs(val_save_path)
    for _, _, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name.endswith(".jpg"):
                continue
            src_path = os.path.join(train_path, name)
            ID = name.split('_')[0]
            ID_path = os.path.join(train_save_path, ID)
            if not os.path.exists(ID_path):
                ID_path = os.path.join(val_save_path, ID)
                os.makedirs(ID_path)
                dst_path = os.path.join(ID_path, name)
                copyfile(src_path, dst_path)
                os.makedirs(os.path.join(train_save_path, ID))
                nums_val += 1
                continue
            dst_path = os.path.join(ID_path, name)
            copyfile(src_path, dst_path)
            nums += 1
    nums_dict['train'] = nums
    nums_dict['val'] = nums_val
    print(f'train: {nums}, val: {nums_val}')
def main():
    parser = argparse.ArgumentParser(description="VeRID dataset pre-process")
    parser.add_argument(
        "--data_dir", default="/home/ha/Downloads/Dataset/VeRi", help="dataset directory", type=str
    )
    
    args = parser.parse_args()
    print(args)
    process_data(dir_path=args.data_dir)

if __name__ == "__main__":
    main()