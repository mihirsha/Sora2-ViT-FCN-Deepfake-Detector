import yaml
from pathlib import Path
import torch

def load_config(config_path: str = "./config/config.yaml"):
    """
    Loads configuration data from a YAML file and processes paths and device settings.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    
    # Processing Paths for Convenience
    BASE_DIR = Path(config['paths']['base_dir'])
    
    # Update the config dictionary with processed Path objects
    config['paths']['BASE_DIR'] = BASE_DIR
    config['paths']['FAKE_VIDEO_DIR'] = BASE_DIR / config['paths']['fake_videos_sub_dir']
    config['paths']['REAL_VIDEO_DIR'] = BASE_DIR / config['paths']['real_videos_sub_dir']
    
    # Set Device
    config['model']['DEVICE'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return config