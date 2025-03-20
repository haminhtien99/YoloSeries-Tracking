import numpy as np
import cv2
import torch
import gc
import time
from typing import Iterable

from ultralytics.models.yolo.detect import DetectionPredictor
from tools.load_yaml import load_yaml
from .sort import Sort
from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker

from .deep_sort import DeepSort

TRACKER_MAP = {'sort': Sort, 'bytetrack': BYTETracker, 'botsort': BOTSORT,
               'deepsort': DeepSort}

class CustomTracker:
    def __init__(self, tracker: str, predictor: DetectionPredictor):
        """
        tracker: path to yaml file config
        """
        cfg = load_yaml(tracker)
        tracker_type = cfg.tracker_type
        if tracker_type not in TRACKER_MAP.keys():
            raise AssertionError(f"Only 'sort', 'botsort', 'bytetrack', 'deepsort'are supported for now, but got '{tracker_type}'")
        self.tracker = TRACKER_MAP[tracker_type](cfg, frame_rate=30)
        self.predictor = predictor

    def update(self, batch):
        img = batch[1]
        results = self.predictor(img)[0]
        boxes = results.boxes.cpu().numpy()
        start = time.time()
        tracks = self.tracker.update(boxes, img[0])
        if len(tracks) == 0:
            results.memory = self._get_memory()
            results.speed['associate'] = 0.0
            return results
        associate_time = (time.time() - start) * 1000
        idx = tracks[:, -1].astype(int)
        valid_indices = idx[idx > -1]
        results = results[valid_indices]

        update_args = {"boxes": torch.as_tensor(tracks[:, :-1])}
        results.update(**update_args)
        results.memory = self._get_memory()
        results.speed['associate'] = associate_time
        return results

    def reset(self):
        self.tracker.reset()
        self._clear_memory()
    def _get_memory(self):
        if self.predictor.device.type == 'cpu':
            memory = 0
        else:
            memory = torch.cuda.memory_reserved()
        return memory/1e9
    def _clear_memory(self):
        gc.collect()
        if self.predictor.device.type == 'cpu':
            return
        else:
            torch.cuda.empty_cache()

def save_img_with_obj(img: np.ndarray,
                      objects: Iterable,
                      img_path: str):
    """
    Save the image with the objects
    """
    if len(objects) == 0:
        return
    for obj in objects:
        x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, str(int(obj[0])), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imwrite(img_path, img)
