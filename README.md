# YOLO detectors and Multi-object tracker on two datasets VisDrone2019 and UAVDT

## 📌 Introduction
This repository contains the code for the YOLO detectors and the Multi-object tracker on two datasets [VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset) and [UAVDT](https://sites.google.com/view/grli-uavdt/%E9%A6%96%E9%A1%B5).

### Supported Detectors:
- Yolov8
- Yolov9
- Yolov10
- Yolo11

### Supported Trackers:
- ByteTrack
- BoT-SORT/BoT-SORT-ReID
- SORT
- DeepSORT
- OC-SORT

### Key Tools & Resources::
- Detectors from [ultralytics](https://github.com/ultralytics/ultralytics)
- Evaluation Tracking [TrackEval](https://github.com/JonathonLuiten/TrackEval)
- Track algorithms: [SORT](https://github.com/abewley/sort), [DeepSORT](https://github.com/nwojke/deep_sort). [OC-SORT](https://github.com/noahcao/OC_SORT), [ByteTrack](https://github.com/ifzhang/ByteTrack), [BoT-SORT](https://github.com/NirAharon/BoT-SORT).

## 🔧 Installation

```
git clone https://github.com/haminhtien99/YoloSeries-Tracking
cd YoloSeries-Tracking
pip install -r requirements.txt
```
## 📂 Dataset preparation
### Object Detection (DET)
Organize your dataset in the following format:
```
dataset_det/
│   |---train_set/
|   |   |---images/     *.jpg ...
|   |   |---labels/     *.txt ... (label_name is the same as image_name)
|   |---val_set/
|   |---test_set/ (optional)
```
Label format (YOLO): 
```
<object-class> <x_center> <y_center> <width> <height> (in range [0, 1])
```
Example `.yaml` config:
```
train: path/to/train_set
val: path/to/val_set
test: path/to/test_set
nc: 1  # number of classes
names: ['class_1', 'class_2', ...]
```
### 📁 Multi-Object Tracking (MOT)
Organize dataset in the following format:
```
dataset_mot/
│   ├── train_set/
│   │   ├── video_name/
│   │   │   ├── img1/                  # *.jpg frames
│   │   │   ├── gt_mot_challenge/
│   │   │   │   └── gt.txt
│   │   │   └── seqinfo.ini

```
`gt.txt` format:
```
<frame>,<id>,<x>,<y>,<w>,<h>,<conf>,<class_id>,<x3D>,<y3D>,<z3D>
-----------------------------------------------------------------------------------------------------------------------------------
Name	                                      Description
-----------------------------------------------------------------------------------------------------------------------------------
<frame>	                The frame index of the video frame

<id>	                  The identity of the target

<x>                     The x coordinate of the top-left corner of the predicted bounding box

<y>	                    The y coordinate of the top-left corner of the predicted object bounding box

<w>                     The width in pixels of the predicted object bounding box

<h>	                    The height in pixels of the predicted object bounding box

<conf>	                The confidence of the predicted bounding box, set to 1

<x3D>,<y3D>,<z3D>       The coordinates in 3D space, set to -1, -1, -1                     
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
### 🗂 Custom Datasets Used
Customized for transport-focused detection/tracking:

| Dataset                                                                                 | Description                                 |
| --------------------------------------------------------------------------------------- | ------------------------------------------- |
| [UAVDT-2024-DET](https://www.kaggle.com/datasets/foryolotrain1/uavdt-2024-det)          | Added stationary vehicles and "van" class   |
| [UAVDT-2024-MOT](https://www.kaggle.com/datasets/foryolotrain1/uavdt-2024-mot)          | MOT format with additional annotations      |
| [VisDrone2019-DET](https://www.kaggle.com/datasets/foryolotrain1/visdrone2019-det-cars) | Filtered to 4 classes: car, truck, bus, van |
| [VisDrone2019-MOT](https://www.kaggle.com/datasets/foryolotrain1/visdrone2019-mot)      | Formatted for tracking                      |


## 🎯 Detector
### Train KD - *In progress* ...
Try knowledge distillation training method for yolo11n, hope it works

Example training config (`yolo11n.yaml`):
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
python -m detector.val \
  --detectors-path path/to/detectors \
  --sub-path dataset_name \
  --model-name yolov8l \
  --data visdrone.yaml \
  --device 0 \
  --batch 1 \
  --project val_results \
  --format pytorch

```
📦 [Download trained detectors](https://www.kaggle.com/datasets/foryolotrain1/yolo-detectors)

## 🧠 ReID Models

Models trained: OSNET_x0_25, ResNet18/34/50, and deepsort_reid.
Exported to TensorRT FP16 (1ms/image on Tesla T4).
See: [`trackers/reid_models/README.md`](trackers/reid_models/README.md)
## 🎥 Tracker

## ⚙️ Configuration
Tracker configs are in [`cfg/trackers/`](cfg/trackers/)
### ▶️ Run Tracking

Track custom video:

```bash
python track_sample.py --tracker sort.yaml --video path/to/your/video --model path/to/your/model

```
Track using dataset & config:

```bash
python track.py --cfg track.yaml
```

### 📊 Evaluation
Prepare ground truth:

```bash
python trackeval/prepare_gt_trackeval.py \
  --BENCHMARK UAVDT \
  --mot_path path/to/UAVDT-2024-MOT
```

Example
```bash
python trackeval/prepare_gt_trackeval.py --BENCHMARK UAVDT --mot_path UAVDT-2024-MOT
```

Directory structure for `results/`
```
results/
├── data/
│   ├── gt/
│   │   └── UAVDT/
│   └── trackers/
│       └── UAVDT/
│           └── val/
│               └── deepsort/
│                   └── data/
│                       └── track_result.txt

...
```
Run eval
```bash
python eval.py \
  --GT_FOLDER results/data/gt/VisDrone \
  --TRACKERS_FOLDER results/data/trackers/VisDrone \
  --TRACKERS_TO_EVAL deepsort \
  --SEQ_INFO uav0000117_02622_v

```

## 📚 References
https://github.com/ultralytics/ultralytics

https://github.com/JonathonLuiten/TrackEval

https://github.com/abewley/sort

https://github.com/mikel-brostrom/Yolov3_DeepSort_Pytorch

https://github.com/nwojke/deep_sort

https://github.com/noahcao/OC_SORT

https://github.com/NirAharon/BoT-SORT

https://github.com/ifzhang/ByteTrack