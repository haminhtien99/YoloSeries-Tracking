# Train ReID model

## Datasets

In this project, use the following datasets

1. VeRi

- [Source](https://github.com/JDAI-CV/VeRidataset)
- The dataset is used for non-commercial purposes
- Downloaded from another source on [kaggle](https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset)
- To train ReID on pytorch, need to run `prepare/veri.py`

<p align="center">
  <img src="images/sample_VeRi.jpg" />
</p>

2. VisDrone - ReID - with 132 ID

- Created by me using the VisDrone2019-MOT-train and VisDrone2019-MOT-val
- Can download from this: [visdrone-reid](https://www.kaggle.com/datasets/foryolotrain1/visdrone-reid)
- Simple dataset, consisting of images of objects taken from above, at night, and at long distance
<p align="center">
  <img src="images/sample_VisDrone.jpg" />
</p>

## ReID Dataset Structure
The dataset follows the standard ReID format, with separate splits for **training**, **validation**, and **testing**. It is organized as follows:

```
pytorch/
├── train_all/
├── val/
├── train/
├── query/
└── gallery/
```
1. **train_all/train/val**
- `train_all`: This directory contains all images, used for training the model on the entire dataset.
- **Split into `train/` and `val/`**:
  - **`train/`**: Contains images for training. These are **all images except the ones used in validation**, ensuring there is **no identity overlap** between training and validation sets.
  - **`val/`**: Used to validate the model during training. This set contains a subset of images from `train_all/`, with each identity being split such that the **first image** of each identity is moved to `val/`, while the remaining images stay in `train/`.
- **No identity overlap**: There is a strict policy of **no identity overlap** between the **training**, **validation**, and **query/gallery** sets. This ensures that the model is evaluated on completely unseen data during testing.


2. **query/gallery**
- **`query/`**: This set contains images from **unseen identities** during training, used to query the model for retrieval tasks.
- **`gallery/`**: Contains images for retrieval, where the model must search through and retrieve matching identities. The **identities** in the `query` and `gallery` sets are also **unseen during training** and are designed to test the model's ability to generalize to new identities.

## Training process

1. Train on VeRi Dataset first
2. Fine-tune the model on VisDrone Dataset

## Code

### Train

```
python train.py --data-dir path/to/reid/dataset --epochs 50 --batch-size 64 --lr0 0.1
```

### Test and evaluate the results

```
python test.py --data-dir path/to/reid/dataset --ckpt-folder folder/to/reid/model
```

## References

https://github.com/JDAI-CV/VeRidataset

https://github.com/layumi/Person_reID_baseline_pytorch

https://github.com/TongJiL/Vehicle-Re-identification-on-VeRi-dataset