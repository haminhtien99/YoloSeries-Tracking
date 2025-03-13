from .distillation_loss import v8DistllationDetectionLoss

from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models import yolo
from copy import copy
class CustomModel(DetectionModel):
    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True,
                 KD_training=True):
        super().__init__(cfg, ch, nc, verbose)
        self.KD_training = KD_training
    def init_criterion(self):
        if not self.KD_training:
            return super().init_criterion(self)
        return v8DistllationDetectionLoss(self)


class CustomTrainer(DetectionTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import CustomTrainer

        args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        trainer = CustomTrainer(overrides=args, KD_training=KD_training)
        trainer.train()
        ```
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, KD_training=True,):
        self.KD_training = KD_training
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=False):
        if not self.KD_training:
            model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        else:
            model = CustomModel(cfg, nc=self.data['nc'], KD_training=self.KD_training,
                                verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        if not self.KD_training:
            self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        else:
            self.loss_names = "box_loss", "cls_loss", "dfl_loss", "KD_box", "KD_cls", "KD_dfl", "KD_feat"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


# args = dict(model="yolo11n.pt", data='cfg/datasets/visdrone-local.yaml', epochs=3, imgsz=320)
# teacher_path = '/home/ha/Downloads/Detectors/visdrone/visdrone/yolov11l/weights/best.pt'
# trainer = CustomTrainer(overrides=args, teacher_path=teacher_path)
# trainer.train()
