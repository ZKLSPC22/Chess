import logging
import sys
import os
from logging.handlers import RotatingFileHandler


def setup_loggers(config):
    '''Configures loggers for the application'''
    log_config = config.get('logging', {})

    # Formatter for all handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. Setup Root Logger (for general application logs)
    app_log_conf = log_config.get('application', {})
    _setup_logger(
        name=None, # Root logger
        level=app_log_conf.get('level', 'INFO'),
        log_file=app_log_conf.get('file', 'logs/app.log'),
        formatter=formatter,
        use_console=app_log_conf.get('console', True)
    )

    # 2. Setup Training Logger
    train_log_conf = log_config.get('training', {})
    _setup_logger(
        name='training',
        level=train_log_conf.get('level', 'INFO'),
        log_file=train_log_conf.get('file', 'logs/training.log'),
        formatter=formatter,
        use_console=train_log_conf.get('console', False)
    )

    # 3. Setup Evaluation Logger
    eval_log_conf = log_config.get('evaluation', {})
    _setup_logger(
        name='evaluation',
        level=eval_log_conf.get('level', 'INFO'),
        log_file=eval_log_conf.get('file', 'logs/evaluation.log'),
        formatter=formatter,
        use_console=eval_log_conf.get('console', False)
    )
    logging.getLogger().info("Logging systems configured successfully.")

def _setup_logger(name, level, log_file, formatter, use_console=False):
    '''Helper to configure an individual logger.'''
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    
    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # File handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5*1024*1024,
        backupCount=3
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if use_console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
