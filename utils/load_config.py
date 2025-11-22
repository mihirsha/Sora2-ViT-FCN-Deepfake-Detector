import yaml
from pathlib import Path
import torch
from utils.logger import get_logger

log = get_logger(__name__)

def load_config(config_path: str = "./config/config.yaml"):
    """
    Loads configuration data from a YAML file and processes paths and device settings.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        log.error(f"Error: Configuration file not found at {config_path}")
        return None
    
    # Set Device
    config['model']['DEVICE'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return config
