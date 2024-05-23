import logging
import os

def setup_logging(log_dir='logs', log_file='app.log', log_level=logging.DEBUG):
    """
    Setup logging configuration.

    Args:
        log_dir (str): Directory where log files will be saved.
        log_file (str): Name of the log file.
        log_level (int): Logging level. Default is logging.DEBUG.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file)

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_path)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

# Example usage
if __name__ == "__main__":
    logger = setup_logging()

    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')
