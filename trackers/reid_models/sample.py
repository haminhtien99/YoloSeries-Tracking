import cv2
import numpy as np

import os
size = 64
images_folder = '/home/ha/Downloads/Dataset/VisDrone-vehicles-ReID/pytorch/query'
ids = os.listdir(images_folder)
sample_image = np.zeros((5*size,8*size,3), dtype=np.uint8)
for i, id in enumerate(ids):
    ls_images = os.listdir(os.path.join(images_folder, id))
    fst_img = ls_images[0]
    img = cv2.imread(os.path.join(images_folder, id, fst_img))
    img = cv2.resize(img, (size, size))
    sample_image[(i//8)* size: (i//8 + 1)* size, (i%8) * size: (i%8 + 1)* size, :] = img

cv2.imwrite('images/sample_VisDrone.jpg', sample_image)