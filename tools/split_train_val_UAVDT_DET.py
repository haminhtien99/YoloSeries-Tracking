import os
import shutil
import random
dir = '/home/ha/Downloads/Dataset/UAVDT-DET/'
train_images = dir + 'train/images/'
val_images = dir + 'val/images/'

train_labels = dir + 'train/labels/'
val_labels = dir + 'val/labels/'
if not os.path.exists(train_images):
    os.makedirs(train_images)
if not os.path.exists(val_images):
    os.makedirs(val_images)
if not os.path.exists(train_labels):
    os.makedirs(train_labels)
if not os.path.exists(val_labels):
    os.makedirs(val_labels)


lss = os.listdir(dir)
labels: list[str] = []
images: list[str] = []
for i in lss:
    if i.split('.')[-1] == 'txt':
        labels.append(i)
    if i.split('.')[-1] == 'jpg':
        images.append(i)
print(len(labels), len(images))

random.shuffle(labels)
for lab in labels[:35000]:
    shutil.move(dir + lab, train_labels)
    name = lab.split('.')[:-1]
    name.append('jpg')
    image = '.'.join(name)
    shutil.move(dir + image, train_images)
print('train moved')
for lab in labels[35000:]:
    shutil.move(dir + lab, val_labels)
    name = lab.split('.')[:-1]
    name.append('jpg')
    image = '.'.join(name)
    shutil.move(dir + image, val_images)
print('val moved')
