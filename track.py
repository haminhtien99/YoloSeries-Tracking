from ultralytics import YOLO
import os
from PIL import Image
from tqdm import tqdm
import argparse

def track_per_model(model_name:str,
                    model_path: str,
                    mot_path: str,
                    track_type: str,
                    splits: str|list[str],
                    imgsz: int,
                    device: str|int|list[int],
                    batch: int,
                    save_img: bool = False,
                    track_folder: str|str = None)-> None:
    """ tracking per model """
    abs_path_runner = os.path.abspath(__file__)
    abs_path_root = os.path.dirname(abs_path_runner)
    benchmark = 'UAVDT' if 'UAVDT' in mot_path else 'VisDrone'
    model = YOLO(model_path)
    track_name = track_type.split('.')[0]
    if not isinstance(splits, (list, tuple)):
        splits = [splits]
    splits_set = [i for i in os.listdir(mot_path) if not i.startswith('README')]
    print(model_name, track_name)
    for spl in splits:
        spl_set = splits_set[0] if spl in splits_set[0] else splits_set[1]
        videos = os.listdir(os.path.join(mot_path, spl_set))
        videos.sort()
        for vid in tqdm(videos, desc=f'{track_name}/{benchmark}/{spl_set}'):
            imgs = os.path.join(mot_path, spl_set, vid, 'img1')
            results = model.track(source=imgs,
                                  device=device,
                                  stream=True,
                                  verbose=False,
                                  tracker=track_type,
                                  persist=True,
                                  imgsz=imgsz,
                                  batch=batch)
            lines = []
            for frame_id, result in enumerate(results):
                for box in result.boxes:
                    bbox = box.xyxy[0].tolist()
                    if box.id is None:
                        continue
                    track_id = box.id.item()
                    conf = box.conf.item()
                    line = f'{frame_id+1},{track_id},{bbox[0]},{bbox[1]},{bbox[2]-bbox[0]},{bbox[3]-bbox[1]},{conf},-1,-1,-1\n'
                    lines.append(line)
                if save_img:
                    # Plot results image
                    im_bgr = result.plot()  # BGR-order numpy array
                    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
                    # Save results to disk
                    track_res_path = os.path.join(abs_path_root, track_folder)
                    if not os.path.exists(track_res_path):
                        os.makedirs(track_res_path)    
                    result_img = os.path.join(track_res_path, f'{frame_id}.jpg')
                    result.save(filename=result_img)
            sub_path = model_path.split('/')[-4]
            full_output_path = os.path.join(abs_path_root,
                                            'TrackEval',
                                            'data',
                                            'trackers',
                                            benchmark,
                                            f'{benchmark}-{spl}',
                                            f'{track_name}-{model_name}-train-{sub_path}',
                                            'data')
            if not os.path.exists(full_output_path):
                os.makedirs(full_output_path)
            file_output = os.path.join(full_output_path, vid + '.txt')
            with open(file_output, 'w') as f:
                f.writelines(lines)
            print(f'save to {file_output}' )
            model.predictor.trackers[0].reset()
def main(args):

    if args.all_weights:
        all_models = os.listdir(os.path.join(args.detectors_path,
                                             args.sub_path, args.sub_path))
        for yolo_name in all_models:
            model_path = os.path.join(args.detectors_path,
                                      args.sub_path, args.sub_path,
                                      yolo_name, 'weights', 'best.pt')
            track_per_model(model_name=yolo_name,
                            model_path=model_path,
                            mot_path=args.mot_path,
                            track_type=args.track_type,
                            splits=args.splits,
                            imgsz=args.imgsz,
                            device=args.device,
                            batch=args.batch,
                            save_img=args.save_img,
                            track_folder=args.track_folder)
    else:
        model_path = os.path.join(args.detectors_path,
                                  args.sub_path, args.sub_path,
                                  args.yolo_name, 'weights', 'best.pt')
        track_per_model(model_name=args.yolo_name,
                        model_path=model_path,
                        mot_path=args.mot_path,
                        track_type=args.track_type,
                        splits=args.splits,
                        imgsz=args.imgsz,
                        device=args.device,
                        batch=args.batch,
                        save_img=args.save_img,
                        track_folder=args.track_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("ultralytics YOLO track parser")
    parser.add_argument('--mot-path', type=str,
                        default='/home/ha/Downloads/Dataset/VisDrone2019-vehicles-MOT',
                        help='path to the MOT-Dataset, UAVDT or VisDrone')
    parser.add_argument('--splits', type=str, default='val',
                        help='track on val, train or [train, val]')
    parser.add_argument('--detectors-path', type=str, default='/home/ha/Downloads/Detectors') # yolo-detectors
    parser.add_argument('--sub-path', type=str, default='uavdt2visdrone',
                        help='folder name, models trained on corresponding dataset')
    parser.add_argument('--all-weights', action='store_true', help='Track using all weights')
    parser.add_argument('--track-type', type=str, default='botsort.yaml',
                        help='type of track, botsort.yaml or bytetrack.yaml')
    parser.add_argument('--yolo-name', type=str, default=None,
                        help='Specific weight to use if all_weights is False')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', type=int, default=320, help='')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-img', action='store_true',
                        help='save results image with track id')
    parser.add_argument('--track-folder', type=str, default=None,
                        help='path to save track results if save_img is True')

    args = parser.parse_args()
    print(f'track type: {args.track_type}')
    print(f'detectors folder: {args.detectors_path}')
    print(f'model trained on dataset: {args.sub_path}')
    print(f'MOT dataset{args.mot_path}')
    if not args.all_weights:
        print(f"Model weight: {args.yolo_name}")
    else:
        print("Using all weights.")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print(f'batch size: {args.batch}')
    if args.save_img:
        print(f"Save image with track id into: {args.track_folder}")
    main(args)
