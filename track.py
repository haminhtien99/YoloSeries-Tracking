import os
import numpy as np
import time

from ultralytics.models import YOLO

from tqdm import tqdm
import argparse
from typing import List

def track_per_video(model: YOLO,
                    imgs: List|str,
                    tracker: str,
                    device: str|List|List[int],
                    stream: bool, # set True if stream
                    imgsz: int,
                    track_folder: str|None,
                    track_txt: str):
    """ tracking per video """
    if not stream:
        start = time.time()
    results = model.track(source=imgs,
                        device=device,
                        stream=False,
                        verbose=False,
                        tracker=tracker,
                        persist=True,
                        imgsz=imgsz,
                        batch=1)
    if not stream:
        summary_time = time.time() - start
        time_per_image = summary_time / len(results)
        print(f'Total time: {summary_time:.3f} seconds, Average time per image: {time_per_image:.3f} seconds')
    # save results to file txt to compute evaluation tracking
    lines = []
    for frame_id, result in enumerate(results):
        boxes = result.boxes.cpu().numpy()
        # print(boxes)
        for box in boxes:
            xyxy = box.xyxy[0]
            if box.id is None:
                continue
            track_id = box.id.item()
            conf = box.conf.item()
            line = f'{frame_id+1},{int(track_id)},{xyxy[0]},{xyxy[1]},{xyxy[2]-xyxy[0]},{xyxy[3]-xyxy[1]},{conf},-1,-1,-1\n'
            lines.append(line)
    with open(track_txt, 'w') as f:
        f.writelines(lines)
    print(f'save to {track_txt}')

    # save result images to track_folder to visualize tracking results
    if track_folder is not None:
        if not os.path.exists(track_folder):
            os.makedirs(track_folder)
        for frame_id, result in enumerate(results):    
            result_img = os.path.join(track_folder, f'{frame_id}.jpg')
            result.save(filename=result_img)
    model.predictor.trackers[0].reset()


def track_per_model(model_name: str,
                    model_path: str,
                    mot_path: str,
                    splits: str|list[str],
                    video: str|None,
                    tracker: str,
                    imgsz: int,
                    device: str|int|list[int],
                    save_img: bool = False,
                    **kwargs)-> None:
    """ tracking per model """
    benchmark = 'UAVDT' if 'UAVDT' in mot_path else 'VisDrone'
    model = YOLO(model_path)
    if not isinstance(splits, (list, tuple)):
        splits = [splits]
    splits_set = [i for i in os.listdir(mot_path) if not i.startswith('README')]
    track_name = tracker.split('.')[0]
    print(model_name, track_name)

    for spl in splits:
        spl_set = splits_set[0] if spl in splits_set[0] else splits_set[1]
        if video is None:
            videos = os.listdir(os.path.join(mot_path, spl_set))
            videos.sort()
        else:
            videos = [video]
        for vid in tqdm(videos, desc=f'{track_name}/{benchmark}/{spl_set}'):
            if save_img:
                track_folder = os.path.join('results', benchmark, f'{benchmark}-{spl}',
                                            f'{track_name}-{model_name}-train-{sub_path}', video)
                print(f'save to {track_folder}')
            else: track_folder = None
            sub_path = model_path.split('/')[-4]
            full_output_path = os.path.join('results',
                                            'data',
                                            'trackers',
                                            benchmark,
                                            f'{benchmark}-{spl}',
                                            f'{track_name}-{model_name}-train-{sub_path}',
                                            'data')
            if not os.path.exists(full_output_path):
                os.makedirs(full_output_path)
            track_txt = os.path.join(full_output_path, vid + '.txt')
            track_per_video(model=model,
                            imgs=os.path.join(mot_path, spl_set, vid, 'img1'),
                            tracker=tracker,
                            device=device,
                            imgsz=imgsz,
                            stream=False,
                            track_folder=track_folder, track_txt=track_txt)


def main(cfg):
    if cfg.all_weights:
        all_models = os.listdir(os.path.join(cfg.detectors_path,
                                             cfg.sub_path, cfg.sub_path))
    else:
        all_models = [cfg.model_name]

    for model_name in all_models:
        model_path = os.path.join(cfg.detectors_path,
                                  cfg.sub_path, cfg.sub_path,
                                  model_name, 'weights', 'best.pt')
        cfg.model_name = model_name
        track_per_model(model_path=model_path,
                        **vars(cfg))


if __name__ == '__main__':
    from tools.load_yaml import load_yaml
    parser = argparse.ArgumentParser("ultralytics YOLO track parser")
    parser.add_argument('--cfg', type=str, default='track.yaml',
                        help='path to cfg file')
    args = parser.parse_args()
    cfg = load_yaml(args.cfg)
    print(f'track type: {cfg.tracker}')
    print(f'detectors folder: {cfg.detectors_path}')
    print(f'model trained on dataset: {cfg.sub_path}')
    print(f'MOT dataset{cfg.mot_path}')
    if not cfg.all_videos:
        print(f'Track on {cfg.video}')
    else:
        cfg.video = None
        print('Track on all videos')
    if not cfg.all_weights:
        print(f"Model weight: {cfg.model_name}")
    else:
        print("Using all weights.")
    print(f"Image size: {cfg.imgsz}")
    print(f"Device: {cfg.device}")

    main(cfg)
