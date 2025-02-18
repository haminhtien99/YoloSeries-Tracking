# copy and edit from
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np
import cv2
import torch
from functools import partial
from pathlib import Path

from typing import Iterable

from .sort import Sort
from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker
from .utils.load_yaml import load_yaml
from .deep_sort import DeepSort

TRACKER_MAP = {'sort': Sort, 'bytetrack': BYTETracker, 'botsort': BOTSORT,
               'deepsort': DeepSort}
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

def on_predict_start(predictor: object, persist: bool = False) -> None:
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool): Whether to persist the trackers if they already exist.

    Raises:
        AssertionError: If the tracker_type is not 'sort'.

    Examples:
        Initialize trackers for a predictor object:
        >>> predictor = SomePredictorClass()
        >>> on_predict_start(predictor, persist=True)
    """
    if hasattr(predictor, "trackers") and persist:
        return
    tracker = predictor.args.tracker
    cfg = load_yaml(tracker)
    tracker_type = cfg.tracker_type
    if tracker_type not in ['sort', 'botsort', 'bytetrack', 'deepsort']:
        raise AssertionError(f"Only 'sort', 'botsort', 'bytetrack', 'deepsort'are supported for now, but got '{tracker_type}'")
    trackers = []
    for _ in range(predictor.dataset.bs):
        
        tracker = TRACKER_MAP[tracker_type](cfg, frame_rate=30)
        trackers.append(tracker)
        if predictor.dataset.mode != "stream":  # only need one tracker for other modes.
            break
    predictor.trackers = trackers
    predictor.vid_path = [None] * predictor.dataset.bs  # for determining when to reset tracker on new video

def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    """
    Postprocess detected boxes and update with object tracking.

    Args:
        predictor (object): The predictor object containing the predictions.
        persist (bool): Whether to persist the trackers if they already exist.

    Examples:
        Postprocess predictions and update with tracking
        >>> predictor = YourPredictorClass()
        >>> on_predict_postprocess_end(predictor, persist=True)
    """
    path, im0s = predictor.batch[:2]

    is_stream = predictor.dataset.mode == "stream"
    for i in range(len(im0s)):
        tracker = predictor.trackers[i if is_stream else 0]
        vid_path = predictor.save_dir / Path(path[i]).name
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path

        boxes = predictor.results[i].boxes.cpu().numpy()
        tracks = tracker.update(boxes, im0s[i])
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        valid_indices = idx[idx > -1]
        predictor.results[i] = predictor.results[i][valid_indices]
        update_args = {"boxes": torch.as_tensor(tracks[:, :-1])}
        predictor.results[i].update(**update_args)

def register_custom_tracker(model: object, persist: bool) -> None:
    """
    Register tracking callbacks to the model for object tracking during prediction.
    Trackers will be registered: Sort

    Args:
        model (object): The model object to register tracking callbacks for.
        persist (bool): Whether to persist the trackers if they already exist.

    Examples:
        Register tracking callbacks to a YOLO model
        >>> model = YOLOModel()
        >>> register_tracker(model, persist=True)
    """
    model.add_callback("on_predict_start", partial(on_predict_start, persist=persist))
    model.add_callback("on_predict_postprocess_end", partial(on_predict_postprocess_end, persist=persist))
