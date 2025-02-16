import yaml
import os

import yaml

class ConfigObject():
    def __init__(self, data):
        for key, val in data.items():
            setattr(self, key, val)
def load_yaml(filename):
    if not os.path.exists(filename):
        filename = os.path.join('tracker', 'cfg', filename)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")
    with open(filename, 'r') as f:
        dict = yaml.safe_load(f)
    return ConfigObject(dict)