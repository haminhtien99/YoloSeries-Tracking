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
- SORT
- DeepSORT
- OC-SORT

In this repository, used these algorithms:
- Detectors from [ultralytics](https://github.com/ultralytics/ultralytics)
- Evaluation Tracking [TrackEval](https://github.com/JonathonLuiten/TrackEval)
- Track algorithms: [SORT](https://github.com/abewley/sort), [DeepSORT](https://github.com/nwojke/deep_sort). [OC-SORT](https://github.com/noahcao/OC_SORT), [ByteTrack](https://github.com/ifzhang/ByteTrack), [BoT-SORT](https://github.com/NirAharon/BoT-SORT).

## Installation

```
git clone https://github.com/haminhtien99/YoloSeries-Tracking
```
Go to cloned folder
```
cd YoloSeries-Tracking
pip install -r requirements.txt
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
train: path/to/train_set
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
### Train KD - In progress ...
Try knowledge distillation training method for yolo11n, hope it works

Before training, prepare the training configuration like this:
```yaml
# Configuration for training yolo model
model: detector/pretrained_weights/yolo11n.pt

data: cfg/datasets/visdrone.yaml # file in cfg/datasets folder or absolute path to dataset configuration
epochs: 3 # number of epochs
batch: 32
imgsz: 320
device: cpu
resume: false
project: detector/train_results
name: yolo11n/exp
patience: 50 # epochs to wait for no observable improvement for early stopping of training


# distillation knowledge training
# path to teacher model if train distillation , otherwise set to null
# teacher: null
teacher:
  path: your/path/to/trained/teacher/best.pt
  temperature: 10.
  lambda_factor: 0.5
```
```bash
python -m detector.train --cfg yolo11n.yaml
```

### Validate

```bash
python detector/val.py --detectors-path path/to/all_detectors --sub-path name/dataset/on/which/models/trained --model-name model/name --data path/to/file.yaml --device 0 -batch 1 --project val_results
```
Example
```bash
python detector/val.py --detectors-path Downloads/yolo-detectors --sub-path visdrone --model-name yolov8l --data visdrone.yaml --device 0 -batch 1 --project val_results
```
### My trained detectors
I train detectors on 2 datasets in 2 different ways. Train separately on 2 datasets or perform training on 2 datasets one after another.
Link to them [yolo-detectors](https://www.kaggle.com/datasets/foryolotrain1/yolo-detectors)

## Tracker
### Track

To track with custom model and sample sequences, run `track_sample.py`

```bash
python track_sample.py --tracker sort.yaml --video path/to/your/video --model path/to/your/model

```
To track with my trained detectors and dataset VisDrone, UAVDT. Please create a configuration file in `cfg` folder, following the [`cfg/track.yaml`](cfg/track.yaml) and run code below

```bash
python track.py --cfg track.yaml
```

### Evaluation tracking
To easily evaluate the tracking results, move the gt folder containing the ground truth information for each dataset into the `results/data/gt` folder (MOT format). Can run the following command to copy the ground truth folder of MOT-dataset to `results/data/gt`

```bash
python trackeval/prepare_gt_trackeval.py --BENCHMARK name_mot_dataset --mot_path path/to/mot_dataset
```

Example
```bash
python trackeval/prepare_gt_trackeval.py --BENCHMARK UAVDT --mot_path UAVDT-2024-MOT
```

Before evaluating, make sure the directory `results` is organized as follows
```
results
|   |---data
|   |   |---gt
|   |   |   |---MOT-dataset-name
|   |   |   |   |---seqmaps
|   |   |   |   |---MOT-dataset-name-val
|   |   |   |   |---MOT-dataset-name-train ...

|   |   |---trackers
|   |   |   |---MOT-dataset-name
|   |   |   |   |---MOT-dataset-name-val
|   |   |   |   |   |---tracker-name
|   |   |   |   |   |   |---data
|   |   |   |   |   |   |   |---track_result.txt ...
...
```
Run eval
```bash
python eval.py --GT-FOLDER results/data/gt/VisDrone --TRACKERS_FOLDER results/data/trackers/VisDrone --TRACKERS_TO_EVAL deepsort --SEQ_INFO uav0000117_02622_v
```

## References
https://github.com/ultralytics/ultralytics

https://github.com/JonathonLuiten/TrackEval

https://github.com/abewley/sort

https://github.com/mikel-brostrom/Yolov3_DeepSort_Pytorch

https://github.com/nwojke/deep_sort

https://github.com/noahcao/OC_SORT
