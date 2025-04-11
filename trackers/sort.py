"""
    Copy and edit from sort github
    https://github.com/abewley/sort


    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import numpy as np
import argparse
from filterpy.kalman import KalmanFilter

from .utils.non_max_suppression import non_max_suppression
from .utils.distance import iou_batch
from .utils.matching import sort_linear_assignment as linear_assignment

from ultralytics.engine.results import Boxes

np.random.seed(0)



def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox):
        # bbox is [x1, y1, x2, y2, score, cls, idx]
        # idx is the index of the detection bbox
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.idx = bbox[-1]
        self.cls = bbox[-2]
        self.score = bbox[-3]

    def update(self, bbox):    
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.idx = bbox[-1]
        self.cls = bbox[-2]
        self.score = bbox[-3]

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        xyxy = convert_x_to_bbox(self.kf.x)
        info = np.array([self.score, self.cls, self.idx])
        return [xyxy[0], info]


def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,7),dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, args, **kwargs):
        """
        Sets key parameters for SORT
        args (Namespace): Command-line arguments containing tracking parameters.

        """
        self.max_age = args.max_age
        self.min_hits = args.min_hits
        self.iou_threshold = args.iou_threshold
        self.max_bbox_overlap = args.max_bbox_overlap
        self.min_score = args.min_score
        self.trackers :list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, boxes: Boxes, img):
        """
        Params:
        det (Boxes): A Boxes object containing bounding boxes and associated information.
        Returns the a similar array, where the last column is the object ID.
        """
        xyxy = boxes.xyxy
        cls = boxes.cls.reshape(-1, 1)
        score = boxes.conf.reshape(-1, 1)
        idx = np.arange(len(cls)).reshape(-1, 1)
        dets = np.concatenate((xyxy, score, cls, idx), axis=1)
        if len(dets) == 0:
            return np.zeros((0, 8))
        tracks = self._update(dets)
        return tracks
    def _update(self, dets = np.empty((0, 7))):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score,cls, idx],[x1,y1,x2,y2,score,cls,idx],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 7)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # delete detections with score under self.min_score
        dets = dets[dets[:, 4] >= self.min_score]

        # Run non-maximum suppression
        bbox_tlwh = self._to_tlwh(dets[:, :4])
        indices = non_max_suppression(bbox_tlwh, self.max_bbox_overlap, dets[:, 4])
        dets = dets[indices]
    
        # get predicted locations from existing trackers.
        tracklets = np.zeros((len(self.trackers), 8))
        to_del = []
        ret = []
        for t, trkl in enumerate(tracklets):
            pos = self.trackers[t].predict()[0]
            trkl[:4] = [pos[0], pos[1], pos[2], pos[3]]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        tracklets = np.ma.compress_rows(np.ma.masked_invalid(tracklets))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, _ = associate_detections_to_trackers(dets, tracklets, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            xyxy, info = trk.get_state()
            if (trk.time_since_update < 1) and \
                (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((xyxy, [trk.id+1], info)).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0,8))
    def reset(self):
        """Resets the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter."""
        self.trackers :list[KalmanBoxTracker] = []
        self.frame_count = 0
    def __str__(self):
        return f'SORT: max_age={self.max_age}, min_hits={self.min_hits}, iou_threshold={self.iou_threshold}, min_score={self.min_score}'
    def _to_tlwh(self, bbox_xyxy: np.ndarray):
        bbox_tlwh = bbox_xyxy.copy()
        bbox_tlwh[:, 0] = bbox_xyxy[:, 0]
        bbox_tlwh[:, 1] = bbox_xyxy[:, 1]
        bbox_tlwh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_tlwh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
        return bbox_tlwh

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args
