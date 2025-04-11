import numpy as np
from typing import List

from .reid_models.feature_extractor import Extractor
from .utils.kalman_filter import KalmanFilterXYAH
from .utils.matching import linear_assignment, matching_cascade
from .utils.detection import Detection
from .utils.non_max_suppression import non_max_suppression
from .utils.distance import NearestNeighborDistanceMetric, iou_distance

from ultralytics.engine.results import Boxes
from ultralytics.utils.ops import xywh2ltwh, empty_like

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class DeepSTrack:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age, det: Detection):

        self.score = det.score
        self.cls = det.cls
        self.idx = det.idx
        self.angle = det.angle # fake argument using to compute iou_distance

        self.features = []
        if det.feature is not None:
            self.features.append(det.feature)

        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self._n_init = n_init
        self._max_age = max_age
        self.state = TrackState.Tentative


    def predict(self, kf: KalmanFilterXYAH):
        """
        Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilterXYAH
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf: KalmanFilterXYAH, detection: Detection):
        """
        Perform Kalman filter measurement update step and update the feature
        cache.

        Args:
            kf (KalmanFilterXYAH): Kalman Filter
            detection (DeepSTrack) : The new track(detection) updated information

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.xyah)
        self.features.append(detection.feature)


        self.idx = detection.idx
        self.cls = detection.cls
        self.score = detection.score
        self.angle = detection.angle

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    @property
    def xyxy(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.tlwh.copy()
        ret[2:] = ret[:2] + ret[2:]
        return ret
    @property
    def xyah(self):
        """
        Get current position in bounding box format `(x_center, y_center, aspect, height)`
        """
        ret = self.tlwh.copy()
        ret[:2] = ret[:2] - ret[2:]/2
        ret[2] /= ret[3]
        return ret
    @property
    def result(self):
        coor = self.xyxy
        return coor.tolist() + [self.track_id, self.score, self.cls, self.idx]

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

INFTY_COST = 1e+5

def gate_cost_matrix(
        kf: KalmanFilterXYAH, cost_matrix, tracks: List[DeepSTrack], detections: List[DeepSTrack], 
        gated_cost=INFTY_COST, only_position=False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.DeepSTrack]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.


    """
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray(
        [det.xyah for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix

class DeepSort(object):
    """
    DeepSort

    Attributes:
        metric (NearestNeighborDistanceMetric): The distance metric used for measurement to track association.
        max_age (int): Maximum number of missed misses before a track is deleted.
        n_init (int): Number of frames that a track remains in initialization phase.
        kf (KalmanFilterXYAH): A Kalman filter to filter target trajectories in image space.
        tracks (List[DeepSTrack]): The list of active tracks at the current time step.
        min_confidence (float): The minimum confidence for using detection object.
        extractor (Extractor): The extractor based ReID model to extract feature from crop box of object.
        max_iou_distance: Maximum IOU distance between bboxes to consider them as the same object
    Methods:


    """
    def __init__(
            self,
            args,
            **kwargs
        ):

        self.min_confidence = args.min_confidence

        self.metric = NearestNeighborDistanceMetric("cosine", args.max_dist, args.nn_budget)
        self.extractor = Extractor(
            args.model_path,
            device=args.device,
            size=tuple(args.imgsz)
        )

        self.max_iou_distance = args.max_iou_distance
        self.max_age = args.max_age
        self.n_init = args.n_init

        self.kf = KalmanFilterXYAH()
        self.tracks: List[DeepSTrack] = []
        self._next_id = 1

        self.nms_max_overlap = args.nms_max_overlap
    def predict(self):  #TODO: run multi_predict
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)
 
    def update(self, boxes: Boxes, ori_img):
        self.height, self.width = ori_img.shape[:2]

        # filter detections
        confidences = boxes.conf
        filter_conf = confidences > self.min_confidence

        confidences = confidences[filter_conf]
        xywh = boxes.xywh[filter_conf]
        tlwh = xywh2ltwh(xywh)
        classes = boxes.cls[filter_conf]
        features = self._get_features(xywh, ori_img)

        detections = self.get_detecions(tlwh, confidences, classes, features)

        # run NMS
        tlwh = np.asarray(tlwh, dtype=np.float32)
        indices = non_max_suppression(tlwh, self.nms_max_overlap, confidences)
        detections = [detections[i] for i in indices]

        # update tracks
        self.predict()
        self._do_update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            res = track.result
            if track.time_since_update == 1:# unmatched tracks
                conf = 0.
                det_idx = -1
                res[-3] = conf
                res[-1] = det_idx

            outputs.append(np.array(res, dtype=np.float32))

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    def _do_update(self, detections: List[Detection]):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.DeepSTrack]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()


        # Init new tracks
        for detection_idx in unmatched_detections:
            det = detections[detection_idx]
            mean, covariance = self.kf.initiate(det.xyah)
            track = DeepSTrack(mean, covariance, self._next_id, self.n_init, self.max_age, det)
            self.tracks.append(track)
            self._next_id += 1

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def get_detecions(self, dets, scores, classes, features):
        num = len(classes)
        if num == 0:
            return []
        dets_idx = np.arange(len(classes))
        return [Detection(tlwh, score, feature, idx, cls)
                for (tlwh, score, cls, idx, feature) in zip(dets, scores, classes, dets_idx, features)]


    def _match(self, detections):

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            matching_cascade(distance_metric=self.gated_metric,
                             max_distance=self.metric.matching_threshold, 
                             cascade_depth=self.max_age,
                             tracks=self.tracks,
                             detections=detections,
                             track_indices=confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            self.iou_association(detections, iou_track_candidates,
                                 unmatched_detections, thresh=self.max_iou_distance)
        matches = matches_a + matches_b
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def gated_metric(self, tracks: List[DeepSTrack], dets: List[Detection]):
        features = np.array([det.feature for det in dets])
        targets = np.array([track.track_id for track in tracks])
        cost_matrix = self.metric.distance(features, targets)
        cost_matrix = gate_cost_matrix(self.kf, cost_matrix, tracks, dets,)
        return cost_matrix

    def iou_association(self, detections: list[Detection], iou_track_ids: list[int], iou_det_ids: list[int], thresh):

        if len(iou_det_ids) == 0 or len(iou_track_ids) == 0:
            return [], list(range(len(iou_track_ids))), list(range(len(iou_det_ids)))


        iou_tracks = [self.tracks[i] for i in iou_track_ids]
        iou_detections = [detections[i] for i in iou_det_ids]
        iou_dist = iou_distance(iou_tracks, iou_detections)
        matches_row_col, unmatched_row, unmatched_col = \
            linear_assignment(iou_dist, thresh, use_lap=True)
        matches, unmatched_tracks, unmatched_detections = [], [], []
        for row, col in matches_row_col:
            track_idx = iou_track_ids[row]
            det_idx = iou_det_ids[col]
            matches.append((track_idx, det_idx))
        for row in unmatched_row:
            track_idx = iou_track_ids[row]
            unmatched_tracks.append(track_idx)
        for col in unmatched_col:
            det_idx = iou_det_ids[col]
            unmatched_detections.append(col)
        return matches, unmatched_tracks, unmatched_detections

    def _xywh_to_xyxy(self, xywh):
        xyxy = empty_like(xywh)
        xy = xywh[..., :2]
        wh = xywh[..., 2:]/2
        xyxy[..., :2] = xy - wh
        xyxy[..., 2:] = xy + wh

        return xyxy.astype(int)

    def _tlwh_to_xyxy(self, tlwh):
        x, y, w, h = tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def _get_features(self, xywh, ori_img):
        xyxy = self._xywh_to_xyxy(xywh)
        im_crops = []
        ori_img = ori_img[..., ::-1]
        for box in xyxy:
            x1, y1, x2, y2 = box
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

    def reset(self):
        self.tracks = []
        self._next_id = 1
