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


def main(args):
    if args.all_weights:
        all_models = os.listdir(os.path.join(args.detectors_path,
                                             args.sub_path, args.sub_path))
    else:
        all_models = [args.model_name]

    for model_name in all_models:
        model_path = os.path.join(args.detectors_path,
                                  args.sub_path, args.sub_path,
                                  model_name, 'weights', 'best.pt')
        args.model_name = model_name
        track_per_model(model_path=model_path,
                        **vars(args))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("ultralytics YOLO track parser")

    parser.add_argument('--mot-path', type=str,
                        default='/home/ha/Downloads/Dataset/VisDrone2019-vehicles-MOT',
                        help='path to the MOT-Dataset, UAVDT or VisDrone')
    parser.add_argument('--splits', type=str, default='val',
                        help='track on val, train or [train, val]')
    parser.add_argument('--all-videos', action='store_true', help='track on all videos')
    parser.add_argument('--video', type=str, default='uav0000117_02622_v',
                        help='name of one video on val [train] mot-path if all_videos is False')

    parser.add_argument('--detectors-path', type=str, default='/home/ha/Downloads/Detectors') # yolo-detectors
    parser.add_argument('--sub-path', type=str, default='visdrone',
                        help='folder name, models trained on corresponding dataset')
    parser.add_argument('--all-weights', action='store_true', help='Track using all weights')
    parser.add_argument('--model-name', type=str, default='yolov8l',
                        help='Specific weight to use if all_weights is False')
    parser.add_argument('--imgsz', type=int, default=640, help='')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--tracker', type=str, default='sort.yaml',
                        help='type of track, sort.yaml, botsort.yaml or bytetrack.yaml')

    parser.add_argument('--save-txt', action='store_true',
                        help='save results to file txt for evaluation')
    parser.add_argument('--save-img', action='store_true',
                        help='save results image with track id')

    args = parser.parse_args()
    print(f'track type: {args.tracker}')
    print(f'detectors folder: {args.detectors_path}')
    print(f'model trained on dataset: {args.sub_path}')
    print(f'MOT dataset{args.mot_path}')
    if not args.all_videos:
        print(f'Track on {args.video}')
    else:
        args.video = None
        print('Track on all videos')
    if not args.all_weights:
        print(f"Model weight: {args.model_name}")
    else:
        print("Using all weights.")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")

    main(args)
