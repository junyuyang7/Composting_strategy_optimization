# 设置日志
import logging
import numpy as np
import random
import os

def set_logger(log_path):
    logger = logging.getLogger()
    # logger.addFilter(lambda record: "findfont" not in record.getMessage())
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_path) 
    file_handler.setFormatter(logging.Formatter("%(asctime)s: %(levelname)s: %(message)s"))
    file_handler.addFilter(lambda record: "findfont" not in record.getMessage())
    logger.addHandler(file_handler)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)