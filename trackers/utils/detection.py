# vim: expandtab:ts=4:sw=4
import numpy as np

class Detection(object):
    """
    DeepSort Detection - This class represents a bounding box detection in a single image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    score : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.
    cls: float
        Class ID of object
    idx: float
        Detection Index in output Detector 

    """

    def __init__(self, tlwh, score, feature, det_idx, cls):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = float(score)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.cls = cls
        self.idx = det_idx
        self.angle = None # fake argument using to compute iou_distance
    @property
    def xyxy(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    @property
    def xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
