# Sample tracking with one video

import os
import argparse
from tqdm import tqdm

from trackers import CustomTracker
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.data.loaders import LoadImagesAndVideos
from ultralytics.utils import ASSETS


def track_per_video(
        imgs: str,
        tracker: CustomTracker,
        track_folder: str|None,
        track_txt: str|None
    ):
    """ tracking per video """
    dataset = LoadImagesAndVideos(path=imgs, batch=1)
    print(f"{'GPU':>11}{'preprocess':>15}{'inference':>15}{'postprocess':>15}{'associate':>15}")
    pbar = tqdm(dataset)
    for i, batch in enumerate(pbar):
        results = tracker.update(batch)

        speed = results.speed
        pbar.set_description(
            ("%11s"*5)
            % (
                f"{results.memory:>10.3g}G",
                f"{speed['preprocess']:>13.2f}ms",
                f"{speed['inference']:>13.2f}ms",
                f"{speed['postprocess']:>13.2f}ms",
                f"{speed['associate']:>13.2f}ms"
            )
        )
        # save results to file txt to compute evaluation tracking
        if track_txt is not None:
            lines = []
            boxes = results.boxes.cpu().numpy()

            for box in boxes:
                xyxy = box.xyxy[0]
                if box.id is None:
                    continue
                track_id = box.id.item()
                conf = box.conf.item()
                line = f'{i+1},{int(track_id)},{xyxy[0]},{xyxy[1]},{xyxy[2]-xyxy[0]},{xyxy[3]-xyxy[1]},{conf},-1,-1,-1\n'
                lines.append(line)

            if i == 0:
                mode = 'w'
            else: mode = 'a'
            with open(track_txt, mode) as f:
                f.writelines(lines)

        # save result images to track_folder to visualize tracking results
        if track_folder is not None:
            if not os.path.exists(track_folder):
                os.makedirs(track_folder)
            dest_img = os.path.join(track_folder, f'{i}.jpg')
            results.save(filename=dest_img)
    if track_txt is not None:
        print(f'save to {track_txt}')



def main(cfg):
    if cfg.save:
        if not os.path.exists(cfg.output):
            os.makedirs(cfg.output)
        track_txt = os.path.join(cfg.output, 'uav0000117_02622_v.txt')
        track_folder = os.path.join(cfg.output, 'seqs')
        if not os.path.exists(track_folder):
            os.makedirs(track_folder)
    else:
        track_folder = None
        track_txt = None
    args = dict(
        model=cfg.model,
        mode='predict',
        batch=1,
        imgsz=cfg.imgsz,
        conf=0.1,   # need low confidence score in some trackers
        verbose=False,
        device=cfg.device,
        save=False
    )
    predictor = DetectionPredictor(overrides=args)
    # warmup
    for img in os.listdir(ASSETS):
        predictor(os.path.join(ASSETS, img))

    tracker = CustomTracker(cfg.tracker, predictor)
    track_per_video(cfg.video, tracker, track_folder, track_txt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("ultralytics YOLO track parser")
    parser.add_argument('--model', type=str,
                        default='detector/train_results/yolo11n/100epochs-none/weights/best.pt')
    parser.add_argument('--video', type=str,
                        default='examples/img1')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--output', type=str, default='runs/track/exp')
    parser.add_argument('--tracker', type=str, default='deepsort.yaml')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='cpu')
    cfg = parser.parse_args()
    print(f'track type: {cfg.tracker}')
    print(f"Image size: {cfg.imgsz}")
    print(f"Device: {cfg.device}")
    if cfg.save:
        print(f'save to {cfg.output}')

    main(cfg)
