'''train yolo series 8, 9, 10, 11'''
import argparse

from tools.load_yaml import load_yaml
from . import DEFAULT_TEACHER_CFG
import yaml
def main(cfg):
    """ main func """

    # overwrite teacher_attr to detector/teacher_cfg.yaml
    teacher_attr = cfg.pop('teacher')
    dataset = load_yaml(cfg['data'], return_dict=True)
    teacher_attr['nc'] = dataset['nc']
    teacher_attr['device'] = cfg['device']

    with open(DEFAULT_TEACHER_CFG, "w") as file:
        yaml.safe_dump(teacher_attr, file, default_flow_style=False)
    from .custom_trainer import CustomTrainer
    trainer = CustomTrainer(overrides=cfg, KD_training=True)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("ultralytics YOLO train parser")
    parser.add_argument('--cfg', type=str, default='yolo11n.yaml',
                        help='configuration file to train yolo model')
    args = parser.parse_args()
    cfg = load_yaml(args.cfg, return_dict=True)
    main(cfg)
