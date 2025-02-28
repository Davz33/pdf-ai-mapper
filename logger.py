import os
import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_level=None):
    """
    Set up a logger with console and file handlers
    
    Args:
        name (str): Logger name
        log_level (str, optional): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                                  Defaults to INFO or value from LOG_LEVEL env var
    
    Returns:
        logging.Logger: Configured logger
    """
    # Get log level from environment variable or use default
    if log_level is None:
        log_level = os.environ.get('LOG_LEVEL', 'INFO')
    
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        f'logs/{name}.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger 