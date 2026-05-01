# src/logger.py

import logging
import sys
import os

def setup_logger(log_file="pipeline.log"):
    """Sets up a global logger with file and console handlers."""
    logger = logging.getLogger("PROTAC_Pipeline")
    logger.setLevel(logging.INFO)

    # Create formatters: Time - Module Name - Level - Message
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

def log_parameters(logger, config):
    """Iterates through config dictionary and logs all training/processing parameters."""
    logger.info("--- PIPELINE CONFIGURATION PARAMETERS ---")
    for section, params in config.items():
        if isinstance(params, dict):
            for key, value in params.items():
                logger.info(f"[{section}] {key}: {value}")
        else:
            logger.info(f"{section}: {params}")
    logger.info("------------------------------------------")