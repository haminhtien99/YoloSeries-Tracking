# Edit from v8DetectionLoss
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import torch
import torch.nn.functional as F
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
from tools.load_yaml import load_yaml
from detector import DEFAULT_TEACHER_CFG


teacher_KD_cfg = load_yaml(DEFAULT_TEACHER_CFG, return_dict=True)
weight, _ = attempt_load_one_weight(teacher_KD_cfg['path'])
teacher = DetectionModel(cfg=weight.yaml, nc=teacher_KD_cfg['nc'], verbose=False)
teacher.load(weight)
teacher.to(teacher_KD_cfg['device'])
teacher.eval()


class v8DistllationDetectionLoss(v8DetectionLoss):
    def __init__(self, student, tal_topk=10):
        super().__init__(student, tal_topk=tal_topk)
        self.temperature = teacher_KD_cfg['temperature']
        self.lambda_factor = teacher_KD_cfg['lambda_factor'] # soft gain loss factor


    def __call__(self, s_preds, batch):
        # Teacher's features
        with torch.no_grad():
            t_preds = teacher(batch['img'].to(next(teacher.parameters()).dtype))

        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(4, device=self.device)
        s_feats = s_preds[1] if isinstance(s_preds, tuple) else s_preds
        t_feats = t_preds[1] if isinstance(t_preds, tuple) else t_preds

        # both models have the same dtype, batch_size, imgsz, anchor_points and stride_tensor
        batch_size = s_feats[0].shape[0]
        dtype = s_feats[0].dtype
        imgsz = torch.tensor(s_feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(s_feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Student's features
        s_feats_concat = torch.cat([xi.view(batch_size, self.no, -1) for xi in s_feats], 2)
        s_pred_distri, s_pred_scores = s_feats_concat.split((self.reg_max * 4, self.nc), 1)
        s_pred_scores = s_pred_scores.permute(0, 2, 1).contiguous()
        s_pred_distri = s_pred_distri.permute(0, 2, 1).contiguous()
        s_pred_bboxes = self.bbox_decode(anchor_points, s_pred_distri)

        # Hard loss
        anc_points = anchor_points * stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            s_pred_scores.detach().sigmoid(),
            (s_pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anc_points,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[1] = self.bce(s_pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                s_pred_distri, s_pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum,
                fg_mask
            )

        # Scale hard loss
        hard_gain = 1 - self.lambda_factor
        loss[0] *= self.hyp.box * hard_gain # box gain
        loss[1] *= self.hyp.cls * hard_gain # cls gain
        loss[2] *= self.hyp.dfl * hard_gain # dfl gain


        t_feats_concat = torch.cat([xi.view(batch_size, self.no, -1) for xi in t_feats], 2)
        t_pred_distri, t_pred_scores = t_feats_concat.split((self.reg_max * 4, self.nc), 1)
        t_pred_distri = t_pred_distri.permute(0, 2, 1)
        t_pred_scores = t_pred_scores.permute(0, 2, 1)

        t_pred_bboxes = self.bbox_decode(anchor_points, t_pred_distri)

        # soft loss

        _, _, _, fg_mask, _ = self.assigner(
            t_pred_scores.sigmoid(),
            (t_pred_bboxes * stride_tensor).type(gt_bboxes.dtype),
            anc_points,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        fg_mask = self.create_mask(fg_mask)
        if fg_mask.sum():
            loss[3] = self.features_distillation_loss(s_feats_concat, t_feats_concat, fg_mask)


        return loss.sum() * batch_size, loss.detach()

    def create_mask(self, mask):
        return mask
    def features_distillation_loss(self, s_feats, t_feats, mask):
        """Calculate distillation loss with temperature scaling."""
        s_feats = (s_feats / self.temperature).permute(0, 2, 1)
        t_feats = (t_feats / self.temperature).permute(0, 2, 1)
        loss = F.kl_div(F.log_softmax(s_feats[mask], dim=1),
                        F.softmax(t_feats[mask], dim=1), reduction='batchmean')
        return loss * (self.temperature ** 2)

if __name__ == '__main__':
    from detector.custom_trainer import CustomTrainer
    args = dict(model="yolo11n.pt", data='cfg/datasets/visdrone-local.yaml', epochs=3, imgsz=320,
                batch=4)
    teacher_path = '/home/ha/Downloads/Detectors/visdrone/visdrone/yolov11l/weights/best.pt'
    trainer = CustomTrainer(overrides=args, KD_training=True)
    trainer._setup_train(1)
    dataloader = trainer.train_loader
    for batch in dataloader:
        print(batch.keys())
        break
    