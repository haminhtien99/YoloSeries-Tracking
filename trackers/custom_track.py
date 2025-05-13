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
from .ocsort import OCSort
from .deep_sort import DeepSort

try:
    import pycuda.driver as cuda
    cuda.init()
except Exception as e:
    print(f"PyCUDA Initialization failed: {e}")



TRACKER_MAP = {'sort': Sort, 'bytetrack': BYTETracker, 'botsort': BOTSORT,
               'deepsort': DeepSort, 'ocsort': OCSort}

class CustomTracker:
    def __init__(self, tracker: str, predictor: DetectionPredictor):
        """
        tracker: path to yaml file config
        """
        cfg = load_yaml(tracker)
        tracker_type = cfg.tracker_type
        if tracker_type not in TRACKER_MAP.keys():
            raise AssertionError(f"Only 'sort', 'botsort', 'ocsort', 'bytetrack', 'deepsort'are supported for now, but got '{tracker_type}'")
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
            results.speed['matching'] = 0.0
            return results
        matching_time = (time.time() - start) * 1000
        idx = tracks[:, -1].astype(int)
        valid_indices = idx[idx > -1]   # hide unmatched track-id
        results = results[valid_indices]

        update_args = {"boxes": torch.as_tensor(tracks[:, :-1])}
        results.update(**update_args)
        results.memory = self._get_memory()
        results.speed['matching'] = matching_time
        return results

    def reset(self):
        self.tracker.reset()
        self._clear_memory()

    def _get_memory(self):  # TODO: get memory when using tenssorrt ??
        if self.predictor.device.type == 'cuda':  # GPU only
            # memory = torch.cuda.memory_reserved()
            free_mem, total_mem = cuda.mem_get_info()
            memory = total_mem - free_mem
            return memory/1e9
        elif self.predictor.device.type == 'mps':
            memory = torch.mps.driver_allocated_memory()
        else:
            memory = 0
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
