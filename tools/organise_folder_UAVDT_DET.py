import os
import shutil
old_dir = '/home/ha/Downloads/Dataset/uavdt-DatasetNinja-remake/'
new_dir = '/home/ha/Downloads/Dataset/UAVDT-DET/'
sub_dir = ['val/', 'train/']
for dir in sub_dir:
    count_img = 0
    count_ann = 0
    images = os.listdir(old_dir + dir + 'images/')
    for img in images:
        shutil.copy(old_dir + dir + 'images/' + img, new_dir)
        count_img += 1
    anns = os.listdir(old_dir + dir + 'labels/')
    for ann in anns:
        shutil.copy(old_dir + dir + 'labels/' + ann, new_dir)
        count_ann += 1
    print(count_ann, count_img)
