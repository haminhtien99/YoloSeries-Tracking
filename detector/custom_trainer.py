import torch
from detector.distillation_loss import v8DistllationDetectionLoss

from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models import yolo
from copy import copy

class CustomModel(DetectionModel):
    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True, teacher_attr=None):
        super().__init__(cfg, ch, nc, verbose)
        if teacher_attr is None:
            return
        self.init_teacher(nc, verbose, teacher_attr)

    def init_teacher(self, nc, verbose, teacher_attr):
        weight, _ = attempt_load_one_weight(teacher_attr['path'])
        self.teacher = DetectionModel(cfg=weight.yaml, nc=nc, verbose=verbose)
        self.teacher.load(weight)
        self.teacher.eval()

        self.teacher.temperature = teacher_attr['temperature']
        self.teacher.lambda_factor = teacher_attr['lambda_factor']

    def init_criterion(self):
        if getattr(self, "teacher", None) is None:
            return super().init_criterion(self)
        return v8DistllationDetectionLoss(self, self.teacher)

    def loss(self, batch, preds=None):
        if getattr(self, 'criterion', None) is None:
            self.criterion = self.init_criterion()
        if getattr(self, 'teacher', None) is None:
            preds = self.forward(batch['img']) if preds is None else preds
            return self.criterion(preds, batch)
        preds = self.forward(batch['img'])
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])
        return self.criterion(preds, teacher_preds, batch)

class CustomTrainer(DetectionTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import CustomTrainer

        args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        trainer = CustomTrainer(overrides=args, teacher=teacher)
        # teacher - simple class with arguments: path, temperature, lambda_factor
        trainer.train()
        ```
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, teacher_attr=None,):
        super().__init__(cfg, overrides, _callbacks)
        if teacher_attr is None:
            return
        self.teacher_attr = teacher_attr

    def get_model(self, cfg=None, weights=None, verbose=False):
        if getattr(self, 'teacher_attr', None) is not None:
            model = CustomModel(cfg, nc=self.data['nc'], teacher_attr=self.teacher_attr,
                                verbose=verbose and RANK == -1)
        else:
            model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        if getattr(self, 'teacher_attr', None) is None:
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
