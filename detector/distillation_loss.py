# Edit from v8DetectionLoss
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import torch
import torch.nn.functional as F
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.loss import v8DetectionLoss

class v8DistllationDetectionLoss(v8DetectionLoss):
    def __init__(self, student, teacher, tal_topk=10):
        super().__init__(student, tal_topk=tal_topk)
        self.teacher = teacher
        self.lambda_factor = teacher.lambda_factor
        self.temperature = teacher.temperature

    def __call__(self, s_preds, t_preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(7, device=self.device)

        s_feats = s_preds[1] if isinstance(s_preds, tuple) else s_preds

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
        s_feats_concatenate = torch.cat([xi.view(batch_size, self.no, -1) for xi in s_feats], 2)
        s_pred_distri, s_pred_scores = s_feats_concatenate.split((self.reg_max * 4, self.nc), 1)
        s_pred_scores = s_pred_scores.permute(0, 2, 1).contiguous()
        s_pred_distri = s_pred_distri.permute(0, 2, 1).contiguous()
        s_pred_bboxes = self.bbox_decode(anchor_points, s_pred_distri)

        # Teacher's features
        t_feats = t_preds[1] if isinstance(t_preds, tuple) else t_preds
        t_feats_concatenate = torch.cat([xi.view(batch_size, self.no, -1) for xi in t_feats], 2)
        t_pred_distri, t_pred_scores = t_feats_concatenate.split((self.reg_max * 4, self.nc), 1)
        t_pred_distri = t_pred_distri.permute(0, 2, 1)
        t_pred_scores = t_pred_scores.permute(0, 2, 1)
        # t_pred_distri.detach_()
        # t_pred_scores.detach_()
        t_pred_bboxes = self.bbox_decode(anchor_points, t_pred_distri)


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
                s_pred_distri, s_pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box # box gain
        loss[1] *= self.hyp.cls # cls gain
        loss[2] *= self.hyp.dfl # dfl gain

        # soft loss
        t_pred_scores.sigmoid_()
        _, _, _, fg_mask, _ = self.assigner(
            t_pred_scores,
            (t_pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anc_points,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        t_sum = max(t_pred_scores.sum(), 1)
        loss[4] = self.bce(s_pred_scores, t_pred_scores.to(dtype)).sum()/t_sum
        state_mask = fg_mask.sum()
        if state_mask:
            loss[3], loss[5] = self.bbox_loss(
                s_pred_distri, s_pred_bboxes, anchor_points, t_pred_bboxes, t_pred_scores, t_sum, fg_mask
            )
        loss[3] *= self.hyp.box * self.lambda_factor
        loss[4] *= self.hyp.cls * self.lambda_factor
        loss[5] *= self.hyp.dfl * self.lambda_factor

        if state_mask:
            loss[6] = self.features_distillation_loss(
                s_feats_concatenate, t_feats_concatenate,
                fg_mask, self.temperature, loss_type='kl_div'
            )
        loss[6] *= self.lambda_factor
        return loss.sum() * batch_size, loss.detach()
    # @torch.jit.script
    def features_distillation_loss(self, s_feats, t_feats, mask, temperature, loss_type='kl_div'):
        """Calculate the features distillation loss between student and teacher models."""
        s_feats = s_feats.permute(0, 2, 1)
        t_feats = t_feats.permute(0, 2, 1)
        scale = 1 / temperature
        input = s_feats[mask] * scale
        target = t_feats[mask] * scale

        if loss_type == 'mse':
            loss = F.mse_loss(input, target)
        elif loss_type == 'bce':
            loss = F.binary_cross_entropy_with_logits(input, target)
        elif loss_type == 'kl_div':
            loss = F.kl_div(F.log_softmax(input, dim=1), F.softmax(target, dim=1), reduction='batchmean')
        return loss * (temperature **2)

