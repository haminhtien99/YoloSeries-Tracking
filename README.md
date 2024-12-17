# YOLO detectors and Multi-object tracker on two datasets VisDrone2019 and UAVDT

## Introduction
This repository contains the code for the YOLO detectors and the Multi-object tracker on two datasets [VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset) and [UAVDT](https://sites.google.com/view/grli-uavdt/%E9%A6%96%E9%A1%B5). The detector supports:
- Yolov8
- Yolov9
- Yolov10
- Yolo11

and the tracker supports:
- ByteTrack
- BoT-SORT

I used the [ultralytics](https://github.com/ultralytics/ultralytics) and [TrackEval](https://github.com/JonathonLuiten/TrackEval) repositories to do this project. Code can be customized for use on other datasets. 

## Installation

python is available

pytorch is available
```
!git clone https://github.com/haminhtien99/YoloSeries-Tracking/tree/master/TrackEval/data # Clone my repository
pip install ultralytics

```
## Dataset preparation
### Dataset-DET
To train yolo-detector, organize dataset in the following format:
```
dataset_det
│   |---train_set
|   |   |---images
|   |   |   |-- *.jpg ...
|   |   |---labels
|   |   |   |-- *.txt ... (label_name is the same as image_name)
|   |---val_set
|   |---test_set (optional)
```
yolo-format for label.txt: 
```
<object-class> <x_center> <y_center> <width> <height> (in range [0, 1])
```
Need to prepare a `.yaml` file to indicate the path to the dataset.
```
train: pathh/to/train_set
val: path/to/val_set
test: path/to/test_set
nc: 1  # number of classes
names: ['class_1', 'class_2', ...]
```
### Dataset-MOT
Organize dataset in the following format:
```
dataset_mot
│   |---train_set
|   |   |---video_name_1
|   |   |   |---img1
|   |   |   |   |---000001.jpg 
|   |   |   |   |---000002.jpg ...
|   |   |   |---gt_mot_challenge
|   |   |   |   |---gt.txt
|   |   |   |---seqinfo.ini
|   |   |---video_name_2 ...
```
file gt.txt looks as follows:
```
<frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<confidence>,<cls_id,<x>,<y>,<z>
-----------------------------------------------------------------------------------------------------------------------------------
Name	                                      Description
-----------------------------------------------------------------------------------------------------------------------------------
<frame_index>	        The frame index of the video frame

<target_id>	            In the GROUNDTRUTH file, the identity of the target is used to provide the  temporal corresponding

<bbox_left>             The x coordinate of the top-left corner of the predicted bounding box

<bbox_top>	            The y coordinate of the top-left corner of the predicted object bounding box

<bbox_width>            The width in pixels of the predicted object bounding box

<bbox_height>	        The height in pixels of the predicted object bounding box

<confidence>	        The confidence of the predicted bounding box, set to 1

<x>,<y>,<z>             The coordinates in 3D space, set to -1, -1, -1                     
```
file seqinfo.ini looks as follows:
```
[Sequence]
name = M0101
imDir = img1
frameRate = 30
seqLength = ...
imWidth = ...
imHeight = ...
imext = .jpg
```
seqLength is the number of frames in the img1 of that sequence folder
### My Datasets, used in this project
I focused on comparing on the object of transportation. So I made some changes compared to the 2 original datasets. On the VisDrone dataset, I only used 4 objects: car, truck, bus and van. On the UAVDT dataset, I added more stationary objects and added 1 object class: van.
- [UAVDT-2024-DET](https://www.kaggle.com/datasets/foryolotrain1/uavdt-2024-det)
- [UAVDT-2024-MOT](https://www.kaggle.com/datasets/foryolotrain1/uavdt-2024-mot)
- [VisDrone2019-DET](https://www.kaggle.com/datasets/foryolotrain1/visdrone2019-det-cars)
- [VisDrone2019-MOT](https://www.kaggle.com/datasets/foryolotrain1/visdrone2019-mot)
## Detector
### Train
```
python train.py --model-weight name_model.pt --data path/to/file.yaml --epochs 100 --batch 8 --device 0 --project path/to/project_name --name name_folder --imgsz 640
```
Example
```
python train.py --model-weight yolov8l.pt --data config/visdrone.yaml --epochs 100 --batch 8 --device 0 --project visdrone --name yolov8l --imgsz 640
```

### Validate

```
python detector/val.py --detectors-path path/to/all_detectors --sub-path name/dataset/on/which/models/trained --model-name model/name --data path/to/file.yaml --device 0 -batch 1 --project val_results
```
Example
```
python detector/val.py --detectors-path Downloads/yolo-detectors --sub-path visdrone --model-name yolov8l --data visdrone.yaml --device 0 -batch 1 --project val_results
```
### My trained detectors
I train detectors on 2 datasets in 2 different ways. Train separately on 2 datasets or perform training on 2 datasets one after another.
Link to them [yolo-detectors](https://www.kaggle.com/datasets/foryolotrain1/yolo-detectors)

## Tracker
### Track

```
python track.py --mot-path path/to/mot_dataset --splits which_part_use_to_track --track-type name_tracker.yaml --detectors-path path/to/all_detectors --sub-path name/dataset/on/which/models/trained --yolo-name model_name --device 0 -batch 1
```
Example
```
python track.py --mot-path /content/UAVDT-2024-MOT --splits train --track-type bytetrack.yaml --detectors-path /content/yolo-detectors --sub-path uavdt --yolo-name yolov8l --device 0 -batch 1
```
You can save images with IDs by running the following command
```
python track.py --mot-path /content/UAVDT-2024-MOT --splits train --track-type bytetrack.yaml --detectors-path /content/yolo-detectors --sub-path uavdt --yolo-name yolov8l --device 0 -batch 1 --save-img --track-foler track_results
```

### Evaluation tracking
Before validating tracker, you need to run the following command to copy the ground truth folder of MOT-dataset to `TrackEval/data/gt`
```
python TrackEval/data/copy_data_to_TrackEval.py --BENCHMARK name_mot_dataset --mot_path path/to/mot_dataset
```

Example
```
python TrackEval/data/copy_data_to_TrackEval.py --BENCHMARK UAVDT --mot_path /content/UAVDT-2024-MOT
```

Run validation
```
python TrackEval/scripts/run_mot_challenge.py --BENCHMARK name_mot_dataset --TRACKERS_TO_EVAL name_tracker --SPLIT_TO_EVAL train --METRICS HOTA CLEAR Identity
```
Read more about TrackEval [here](https://github.com/JonathonLuiten/TrackEval)