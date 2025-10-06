import logging
import sys
from datetime import datetime
import os
from typing import Optional, Any

def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """
    Configure structured logging for the application.

    This function sets up a logger with both file and console handlers, 
    creating a log file in the 'logs' directory and configuring logging 
    to both file and standard output.

    Args:
        log_level (int, optional): The logging level to set. 
            Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger for the application.
    """
    # Create a custom formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a file handler
    log_filename = os.path.join(logs_dir, f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    
    # Create a stream handler for console output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    
    # Get the root logger
    logger = logging.getLogger('sound_realty_api')
    logger.setLevel(log_level)
    
    # Clear any existing handlers to prevent duplicate logs
    logger.handlers.clear()
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger
