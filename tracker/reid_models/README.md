# Train ReID model for tracking vehicle

## Dataset VeRi

https://github.com/JDAI-CV/VeRidataset
The dataset is used for non-commercial purposes

Downloaded from another source on kaggle
https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset

## Structure of the dataset

1. "image_query/". This dir contains 1678 images as queries.
2. "image_test/". This dir contains 11579 images for testing.
3. "image_train/". This dir contains 37778 images for training.
4. "name_query.txt". It lists all query file names.
5. "name_test.txt". It lists all test file names.
6. "name_train.txt". It lists all train file names.
7. "test_track.txt". It records all test tracks. Each track contrains about 6 images of the same vehicle captured by one camera at a time.
8. "gt_image.txt". It lists the ground truths of each query in each line.
9. "jk_image.txt". It lists the junk images of each query in each line. The junk images are the images with the same camera ID to the query.
10. "train_label.xml". It lists the labels, e.g., vehicle ID, camera ID, color, type, of the training images.
11. "test_label.xml". It lists the labels of all test images.
12. "list_color.txt". It lists the numbers of colors in the xml files.
13. "list_type.txt". It lists the numbers of tpyes in the xml files.
14. "camera_ID.txt". It lists the IDs of all 20 cameras.
15. "camera_Dist.txt". It records the distances between each pair of cameras.
16. "YongtaiPoint_Google.jpg". It is the map with the camera locations.

## Prepare the dataset for training model ReID
Run `prepare_vehicle_reid.py`

## Train

Run `train.py`

## Test and evaluate the results

Run `test.py`

## References

https://github.com/JDAI-CV/VeRidataset

https://github.com/layumi/Person_reID_baseline_pytorch

https://github.com/TongJiL/Vehicle-Re-identification-on-VeRi-dataset