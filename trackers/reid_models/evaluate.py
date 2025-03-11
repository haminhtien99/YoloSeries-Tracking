import os
import torch
import numpy as np
import argparse
import torch.nn.functional as F
from collections import defaultdict
def build_dist(features_1: torch.Tensor, features_2: torch.Tensor, metric_distance='cosine'):
    if metric_distance == 'cosine':
        features_1 = F.normalize(features_1, p=2, dim=1)
        features_2 = F.normalize(features_2, p=2, dim=1)
        dist = 1 - torch.mm(features_1, features_2.t())
    elif metric_distance == 'euclidean':
        m, n = features_1.size(0), features_2.size(0)
        dist = (torch.pow(features_1, 2).sum(dim=1, keepdim=True).expand(m, n)
                + torch.pow(features_2, 2).sum(dim=1, keepdim=True).expand(n, m).t())
        dist.addmm_(features_1, features_2.t(), beta=1, alpha=-2)

    return dist.cpu().numpy()

def evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """
    Fast ReID evaluation
    Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        matches = (g_pids[order] == q_pid).astype(np.int32)
        raw_cmc = matches[keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc, all_AP, all_INP

def evaluate(features, metric_distance='cosine', max_rank=50):
    gallery_cameras = features['gc'].cpu().numpy()
    gallery_features = features['gf']
    gallery_labels = features['gl'].cpu().numpy()
    query_cameras = features['qc'].cpu().numpy()
    query_features = features['qf']
    query_labels = features['ql'].cpu().numpy()

    dist_matrix = build_dist(query_features, gallery_features, metric_distance=metric_distance)
    cmc, all_AP, all_INP = evaluate_rank(dist_matrix, query_labels, gallery_labels, query_cameras, gallery_cameras,
                                         max_rank)
    mAP = np.mean(all_AP) * 100
    mINP = np.mean(all_INP) * 100
    Rank1 = cmc[0] * 100
    Rank5 = cmc[4] * 100
    Rank10 = cmc[9] * 100

    print(f'Rank@1: {Rank1:.1f}, Rank5: {Rank5:.1f}, Rank10: {Rank10:.1f}')
    print(f'mAP: {mAP:.1f}, mINP: {mINP:.1f}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the features output')
    parser.add_argument(
        '--ckpt-folder', type=str, default='veri-128-128',
        help='checkpoint folder of feature')
    parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'euclidean'],
                        help='metric distance to compute distance')
    args = parser.parse_args()
    features_path = os.path.join('checkpoint', args.ckpt_folder, 'features.pth')
    try:
        features = torch.load(features_path)
    except:
        print('Error loading features')
    evaluate(features, metric_distance=args.metric)
