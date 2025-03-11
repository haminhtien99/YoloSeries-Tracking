import yaml
import os
from typing import Any, Dict

class ConfigObject:
    """Recursive configuration object that allows attribute-style access"""
    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)
    
    def __repr__(self) -> str:
        return str(self.__dict__)

def load_yaml(filename: str, return_dict=False) -> ConfigObject|Dict:
    """Load YAML configuration file into ConfigObject
    
    Args:
        filename (str): Path to YAML config file
        
    Returns:
        ConfigObject: Nested configuration object
    """
    # Try possible file paths
    possible_paths = [
        filename,
        os.path.join('cfg', filename),
        os.path.join('cfg', 'trackers', filename),
        os.path.join('cfg', 'detectors', filename)
    ]
    
    for filepath in possible_paths:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
            if return_dict:
                return config_dict
            return ConfigObject(config_dict)
    
    raise FileNotFoundError(
        f"Config file not found. Tried: {possible_paths}"
    )