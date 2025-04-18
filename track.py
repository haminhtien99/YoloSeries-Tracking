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
    avg_preprocess, avg_inference, avg_postprocess, avg_associate = 0., 0., 0., 0.
    dataset = LoadImagesAndVideos(path=imgs, batch=1)
    print(f"{'GPU':>11}{'preprocess':>15}{'inference':>15}{'postprocess':>15}{'associate':>15}")
    pbar = tqdm(dataset)
    for i, batch in enumerate(pbar):
        results = tracker.update(batch)

        preprocess = results.speed['preprocess']
        avg_preprocess += preprocess
        inference = results.speed['inference']
        avg_inference += inference
        postprocess = results.speed['postprocess']
        avg_postprocess += postprocess
        associate = results.speed['associate']
        avg_associate += associate

        pbar.set_description(
            ("%11s"*5)
            % (
                f"{results.memory:>10.3g}G",
                f"{preprocess:>13.2f}ms",
                f"{inference:>13.2f}ms",
                f"{postprocess:>13.2f}ms",
                f"{associate:>13.2f}ms"
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

    tracker.reset()
    avg_preprocess /= (i + 1)
    avg_inference /= (i+1)
    avg_postprocess /= (i+1)
    avg_associate /= (i +1)
    return {'preprocess': avg_preprocess, 'inference': avg_inference, 'postprocess': avg_postprocess, 'associate':avg_associate}

def track(
        model_name: str,
        model_path: str,
        mot_path: str,
        splits: str|list[str],
        video: str|None,
        tracker_cfg: str,
        imgsz: int,
        device: str|int,
        save_img=False,
        save_txt=True,
        conf=0.1,
        **kwargs
    )-> None:
    """
    tracking per model in MOT dataset

    Args:
        model_name: yolo name- yolov8l, yolov8m, ...
        model_path: path to weights
        mot_path: dataset MOT
        splits: train or val or ['train', 'val']
        video: specific video in mot_path/splits 
        imgsz: image size
        device: 'cpu' or 0, 1, ...
        save_img: save image with track id
        save_txt: save track results for evaluating
        conf: confidence score of detected object

    """
    benchmark = 'UAVDT' if 'UAVDT' in mot_path else 'VisDrone'
    args = dict(
        model=model_path,
        name=model_name,
        mode='predict',
        batch=1,
        imgsz=imgsz,
        conf=conf,
        verbose=False,
        device=device,
        save=False
    )
    predictor = DetectionPredictor(overrides=args)

    # warmup
    for img in os.listdir(ASSETS):
        predictor(os.path.join(ASSETS, img))

    tracker = CustomTracker(tracker_cfg, predictor)
    if not isinstance(splits, (list, tuple)):
        splits = [splits]
    splits_set = [i for i in os.listdir(mot_path) if not i.startswith('README')]
    track_name = tracker_cfg.split('.')[0]
    print(model_name, track_name)
    times = {'preprocess': [], 'inference': [], 'postprocess': [], 'associate': []}
    for spl in splits:
        spl_set = splits_set[0] if spl in splits_set[0] else splits_set[1]
        if video is None:
            videos = os.listdir(os.path.join(mot_path, spl_set))
            videos.sort()
        else:
            videos = [video]

        for vid in videos:
            print(f'{spl_set}/{vid}')
            if save_img:
                track_folder = os.path.join('results', benchmark, f'{benchmark}-{spl}',
                                            f'{track_name}-{model_name}-train-{sub_path}',
                                            video)
                print(f'save to {track_folder}')
            else: track_folder = None

            if save_txt:
                sub_path = model_path.split('/')[-4]
                full_output_path = os.path.join(
                    'results',
                    'data',
                    'trackers',
                    benchmark,
                    f'{benchmark}-{spl}',
                    f'{track_name}-{model_name}-train-{sub_path}',
                    'data'
                )
                if not os.path.exists(full_output_path):
                    os.makedirs(full_output_path)
                track_txt = os.path.join(full_output_path, vid + '.txt')

            else: track_txt = None

            res = track_per_video(
                imgs=os.path.join(mot_path, spl_set, vid, 'img1'),
                tracker=tracker,
                track_folder=track_folder,
                track_txt=track_txt
            )
            for key in res.keys():
                times[key].append(res[key])
        if save_txt:
            print(f'saved text to{full_output_path}')
    return times

def main(cfg):
    if cfg.all_weights:
        all_models = os.listdir(os.path.join(cfg.detectors_path,
                                             cfg.sub_path, cfg.sub_path))
    else:
        all_models = [cfg.model_name]

    for model_name in all_models:
        if cfg.format == 'onnx':
            weight = 'best.onnx'
        elif cfg.format == 'engine':
            weight = 'best.engine'
        else:
            weight = 'best.pt'
        model_path = os.path.join(cfg.detectors_path,
                                  cfg.sub_path, cfg.sub_path,
                                  model_name, 'weights', weight)
        if not os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                pt_weight = model_path.split('.')[0] + '.pt'
                model = YOLO(pt_weight)
                model.export(format=cfg.format)
            except:
                print(f'{model_path} not exists')
        cfg.model_name = model_name
        times = track(model_path=model_path, **vars(cfg))

        number_vid = len(times['preprocess'])
        time1 = sum(times['preprocess'])/number_vid
        time2 = sum(times['inference'])/number_vid
        time3 = sum(times['postprocess'])/number_vid
        time4 = sum(times['associate'])/number_vid
        print(f"{'Average':<11}{time1:13.2f}ms{time2:13.2f}ms{time3:13.2f}ms{time4:13.2f}ms")

if __name__ == '__main__':
    from tools.load_yaml import load_yaml
    parser = argparse.ArgumentParser("ultralytics YOLO track parser")
    parser.add_argument('--cfg', type=str, default='track.yaml',
                        help='path to cfg file')
    args = parser.parse_args()
    cfg = load_yaml(args.cfg)
    print(f'track type: {cfg.tracker_cfg}')
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
