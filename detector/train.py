'''train yolo series 8, 9, 10'''
from ultralytics import YOLO
import argparse

def main(args):
    """ main func """

    model = YOLO(model=args.model_weight)
    model.train(data=args.data,
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                patience=50,  # epochs to wait for no observable improvement for early stopping of training
                device=args.device,
                name=args.name,
                project=args.project
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser("ultralytics YOLO train parser")

    parser.add_argument('--model-weight', type=str, default='detector/pretrained_weights/yolov8n.pt', help='')
    parser.add_argument('--data', type=str, default='data-config/visdrone-local.yaml', help='')
    parser.add_argument('--epochs', type=int, default=1, help='')
    parser.add_argument('--batch', type=int, default=1, help='')
    parser.add_argument('--imgsz', type=int, default=320, help='')
    parser.add_argument('--device', type=str, default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--name', type=str, default='yolov8/exp', help='name of ouput folder')
    parser.add_argument('--project', type=str, default='detector/ultralytics/runs', help='path to save project files and results')
    args = parser.parse_args()

    main(args)
