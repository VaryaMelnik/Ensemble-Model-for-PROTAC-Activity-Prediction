# config.py

import yaml
import logging

logger = logging.getLogger("PROTAC_Pipeline.config")

def load_config(config_path="config.yaml"):
    """
    Safely loads the YAML configuration file.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded successfully from {config_path}")
            return config
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        raise
    except Exception as e:
        logger.error(f"Error parsing config file: {e}")
        raise