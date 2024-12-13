from ultralytics import YOLO
import os
from PIL import Image
from tqdm import tqdm
detector_path = '/home/ha/Downloads/Detectors'
splits_detector = [i for i in os.listdir(detector_path) if not i.endswith('zip')]
full_detectors = []
for spl in splits_detector:
    versions_yolo = os.listdir(os.path.join(detector_path, spl))
    for ver in versions_yolo:
        full_detectors.append(os.path.join(spl, ver))

version = full_detectors[0].split('/')[1]
src_mot = full_detectors[0].split('/')[0]
detector_path = os.path.join(detector_path, full_detectors[0], 'weights', 'best.pt')
model = YOLO(detector_path)

mots_path = {'visdrone': '/home/ha/Downloads/Dataset/VisDrone2019-vehicles-MOT',
             'uavdt': '/home/ha/Downloads/Dataset/UAVDT-2024-MOT'}
benchmarks = {'visdrone': 'VisDrone',
              'uavdt': 'UAVDT'}
mot_path = mots_path[src_mot]
benchmark = benchmarks[src_mot]

splits = [i for i in os.listdir(mot_path) if not i.startswith('README')]
trackers_results = f'/home/ha/projects/YoloSeries-Tracking/TrackEval/data/trackers/{benchmark}/'
track_type = 'botsort.yaml'
track_name = track_type.split('.')[0]
for spl in splits:
    videos = os.listdir(os.path.join(mot_path, spl))
    videos.sort()
    for vid in tqdm(videos, desc=f'{track_name}/{benchmark}/{spl}'):
        imgs = os.path.join(mot_path, spl, vid, 'img1')
        results = model.track(source=imgs,
                              device='cpu',
                              stream=True,
                              verbose=False,
                              tracker=track_type,
                              persist=True)
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
            # # Plot results image
            # im_bgr = result.plot()  # BGR-order numpy array
            # im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
            # # Save results to disk
            # result.save(filename=f"/home/ha/projects/YoloSeries-Tracking/track_results/results{frame_id}.jpg")
        if 'train' in spl:
            full_output_path = os.path.join(trackers_results, f'{benchmark}-train', f'{track_name}-{version}', 'data')
        else:
            full_output_path = os.path.join(trackers_results, f'{benchmark}-val', f'{track_name}-{version}', 'data')
        if not os.path.exists(full_output_path):
            os.makedirs(full_output_path)
        file_output = os.path.join(full_output_path, vid + '.txt')
        with open(file_output, 'w') as f:
            f.writelines(lines)
        model.predictor.trackers[0].reset()
    