from .distillation_loss import v8DistllationDetectionLoss

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models import yolo
from copy import copy

class CustomModel(DetectionModel):
    def init_criterion(self):
        return v8DistllationDetectionLoss(self)

class CustomTrainer(DetectionTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import CustomTrainer

        args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        trainer = CustomTrainer(overrides=args)
        trainer.train()
        ```
    """

    def get_model(self, cfg=None, weights=None, verbose=False):
        model = CustomModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "KD_feat"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


# args = dict(model="yolo11n.pt", data='cfg/datasets/visdrone-local.yaml', epochs=3, imgsz=320)
# teacher_path = '/home/ha/Downloads/Detectors/visdrone/visdrone/yolov11l/weights/best.pt'
# trainer = CustomTrainer(overrides=args)
# trainer.train()
