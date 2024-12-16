from ultralytics import YOLO
import argparse
import os
from tqdm import tqdm
import numpy as np
import csv
os.environ['YOLO_VERBOSE'] = 'False'
def val_per_version(model_name: str,
                    model_path: str,
                    project: str,
                    data: str,
                    imgsz: int,
                    device: str) -> dict:
    """ validation per version """
    model = YOLO(model=model_path)
    metrics = {}
    results = model.val(data=data,
                        imgsz=imgsz,
                        device=device,
                        project=project,
                        name=model_name)
    metrics['name'] = model_name
    metrics['mAP50'] = np.around(results.box.map50, 3)
    metrics['mAP75'] = np.around(results.box.map75, 3)
    metrics['mAP50-95'] = np.around(results.box.map, 3)
    metrics['mAP50-car'] = np.around(results.box.class_result(0)[2], 3)
    metrics['mAP50-truck'] = np.around(results.box.class_result(1)[2], 3)
    metrics['mAP50-bus'] = np.around(results.box.class_result(2)[2], 3)
    metrics['mAP50-van'] = np.around(results.box.class_result(3)[2], 3)
    return metrics
def main(args):
    """ main func """
    abs_path_runner = os.path.abspath(__file__)
    abs_path_root = os.path.dirname(abs_path_runner)
    results_list = []
    project = os.path.join(abs_path_root, args.project, 'train-' + args.sub_path)
    if not os.path.exists(project):
        os.makedirs(project)
        print(f'Validation folder {project} created')
    if args.all_weights:
        all_models = os.listdir(os.path.join(args.detectors_path,
                                             args.sub_path, args.sub_path,))
        for yolo_name in tqdm(all_models, desc=f'{args.sub_path}'):
            print(yolo_name)
            model_path = os.path.join(args.detectors_path,
                                      args.sub_path, args.sub_path,
                                      yolo_name, 'weights', 'best.pt')
            metrics = val_per_version(model_name=yolo_name,
                                      model_path=model_path,
                                      project=project,
                                      data=args.data,
                                      imgsz=args.imgsz,
                                      device=args.device)
            results_list.append(metrics)
    else:
        model_path = os.path.join(args.detectors_path,
                                  args.sub_path, args.sub_path,
                                  args.model_name, 'weights', 'best.pt')
        metrics = val_per_version(model_name=args.model_name,
                                  model_path=model_path,
                                  project=project,
                                  data=args.data,
                                  imgsz=args.imgsz,
                                  device=args.device)
        results_list.append(metrics)
    with open(os.path.join(project, 'results.csv'), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=results_list[0].keys())
        writer.writeheader()
        writer.writerows(results_list)
    absolute_path_project = os.path.abspath(project)
    print(f"Results saved to {absolute_path_project}/results.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("ultralytics YOLO validation parser")

    parser.add_argument('--detectors-path', type=str, default='/home/ha/Downloads/Detectors')
    parser.add_argument('--sub-path', type=str, default='uavdt',
                        help='folder name, models trained on corresponding dataset')
    parser.add_argument('--all-weights', action='store_true', help='Track using all weights')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Specific weight to use if all_weights is False')
    parser.add_argument('--data', type=str, default='data-config/uavdt.yaml', help='')
    parser.add_argument('--batch', type=int, default=-1, help='')
    parser.add_argument('--imgsz', type=int, default=320, help='')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--name', type=str, default='uavdt/exp1', help='name of ouput folder')
    parser.add_argument('--project', type=str, default='val_results',
                        help='path to save project files and results')
    args = parser.parse_args()
    print(f'validation dataset: {args.data}')
    print(f'detectors folder: {args.detectors_path}')
    print(f'model trained on dataset: {args.sub_path}')
    if not args.all_weights:
        print(f"Model name: {args.model_name}")
    else:
        print("Using all weights.")
    print(f'imgsz: {args.imgsz}')
    print(f'device: {args.device}')
    print(f'output folder: {args.project}')
    main(args)
