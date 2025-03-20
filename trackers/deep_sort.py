import numpy as np
import torch

from .reid_models.feature_extractor import Extractor

from .deepsort.nn_matching import NearestNeighborDistanceMetric
from .deepsort.detection import Detection
from .deepsort.multiple_track import Tracker

from .utils.non_max_suppression import non_max_suppression

from ultralytics.engine.results import Boxes
class DeepSort(object):
    def __init__(
            self,
            args,
            **kwargs
        ):
        self.min_confidence = args.min_confidence
        self.nms_max_overlap = args.nms_max_overlap

        self.extractor = Extractor(
            args.name,
            args.model_path,
            use_cuda=args.use_cuda,
            size=tuple(args.imgsz)
        )

        max_cosine_distance = args.max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, args.max_iou_distance, args.max_age, args.n_init)


    def update(self, boxes: Boxes, ori_img):
        self.height, self.width = ori_img.shape[:2]

        # generate detections
        confidences = boxes.conf
        filter_conf = confidences > self.min_confidence
        confidences = confidences[filter_conf]
        bbox_xywh = boxes.xywh[filter_conf]
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        classes = boxes.cls[filter_conf]
        features = self._get_features(bbox_xywh, ori_img)
        detections = [Detection(bbox_tlwh[i], conf, features[i], [classes[i], conf, i])
                      for i,conf in enumerate(confidences)]

        # run on non-maximum supression
        bbox_tlwh = np.asarray(bbox_tlwh, dtype=np.float32)
        indices = non_max_suppression(bbox_tlwh, self.nms_max_overlap, confidences)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            if track.time_since_update == 1:# unmatched tracks
                conf = 0.
                det_id = -1
                cls = track.add_infor[0]
            else: cls, conf, det_id = track.add_infor
            outputs.append(np.array([x1, y1, x2, y2, track_id, conf, cls, det_id], dtype=np.float32))

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features
    def reset(self):
        self.tracker.reset()

if __name__ == '__main__':
    from trackers.utils.load_yaml import load_yaml
    args = load_yaml("trackers/cfg/deepsort.yaml")
    deepsort = DeepSort(args)
    import cv2
    ori_img = cv2.imread("/home/ha/Downloads/Dataset/VisDrone2019-vehicles-MOT/VisDrone2019-MOT-val/uav0000117_02622_v/img1/0000001.jpg")
    
    # Create sample detection boxes
    bboxes = torch.tensor([
        [443,754,443 + 147,754+156, 0.9, 0],
        [1490,451,1490+147,451+93, 0.7, 0],
        [1235,388,1235+94,388+68, 0.4, 0]
    ])
    # Create Boxes object
    det = Boxes(bboxes, ori_img)
    det = det.cpu().numpy()
    # Run the update method
    tracks = deepsort.update(det, ori_img)
    tracks = deepsort.update(det, ori_img)
    tracks = deepsort.update(det, ori_img)
    bboxes = torch.tensor([
        [443,754,443 + 147,754+156, 0.9, 0],
        [1490,451,1490+147,451+93, 0.7, 0]
    ])
    det = Boxes(bboxes, ori_img)
    det = det.cpu().numpy()
    tracks = deepsort.update(det, ori_img)


    print(tracks)
