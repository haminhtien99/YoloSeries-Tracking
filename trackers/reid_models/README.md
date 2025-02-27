# Train ReID model

## Datasets

In this project, use the VeRi dataset


- [Source](https://github.com/JDAI-CV/VeRidataset)
- The dataset is used for non-commercial purposes
- Downloaded from another source on [kaggle](https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset)
- To train ReID on pytorch, need to run `prepare/veri.py`

<p align="center">
  <img src="images/sample_VeRi.jpg" />
</p>

## ReID Dataset Structure
The dataset follows the standard ReID format, with separate splits for **training**, **validation**, and **testing**. It is organized as follows:

```
pytorch/
├── train/
├── query/
└── gallery/
```
1. **train**
- **No identity overlap**: There is a strict policy of **no identity overlap** between the **train**, **query/gallery** sets. This ensures that the model is evaluated on completely unseen data during testing.

2. **query/gallery**
- **`query/`**: This set contains images from **unseen identities** during training, used to query the model for retrieval tasks.
- **`gallery/`**: Contains images for retrieval, where the model must search through and retrieve matching identities. The **identities** in the `query` and `gallery` sets are also **unseen during training** and are designed to test the model's ability to generalize to new identities.

## Training process - Updating ...

## Code

### Train

```
python train.py --config path/to/config/file.yml
```

### Test and evaluate the results

```
python test.py --config path/to/config/file.yml
```

## References

https://github.com/JDAI-CV/VeRidataset

https://github.com/layumi/Person_reID_baseline_pytorch

https://github.com/TongJiL/Vehicle-Re-identification-on-VeRi-dataset

https://github.com/JDAI-CV/fast-reid