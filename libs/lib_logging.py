import logging
import os


def check_if_dir_exists(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        print(f'Log directory does not exist, new folder created: {dir_path}')


def spawn_logger(name, formatter='%(asctime)s - %(message)s'):
    log_directory = 'log'
    check_if_dir_exists(log_directory)
    path = os.path.join(log_directory, name)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(path)
    ch = logging.StreamHandler()

    formatter = logging.Formatter(formatter)
    file_handler.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(file_handler)

    return logger
