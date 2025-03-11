'''train yolo series 8, 9, 10, 11'''
import argparse
from .custom_trainer import CustomTrainer
from tools.load_yaml import load_yaml
def main(cfg):
    """ main func """
    teacher_attr = cfg.pop('teacher')
    trainer = CustomTrainer(overrides=cfg, teacher_attr=teacher_attr)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("ultralytics YOLO train parser")
    parser.add_argument('--cfg', type=str, default='yolo11n.yaml',
                        help='configuration file to train yolo model')
    args = parser.parse_args()
    cfg = load_yaml(args.cfg, return_dict=True)
    main(cfg)
