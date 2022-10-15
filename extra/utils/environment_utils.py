import os
import re
import torch
import logging
import colorlog

from recbole.utils.utils import get_local_time, ensure_dir
from colorama import init


def auto_allocate_gpu(config, logger=None, max_num_device=2, max_gpu_usage_allowed=1.):
    r"""Automaticlly allocate GPU. 

        If current GPU is busy, the device will be change to next GPU based on index. 

        If all GPU is busy, it will raise `SystemExit`.
    """
    if config['use_gpu'] is True and max_num_device > 1:
        device = config['device']
        memory_usage = (torch.cuda.memory_allocated(device) + torch.cuda.memory_reserved(device))/1024**3
        # if usage of current GPU memory bigger than 1G, change to next gpu
        while memory_usage > max_gpu_usage_allowed and (device.index + 1) % max_num_device != config['gpu_id']:
            device = torch.device((device.index + 1) % max_num_device)
            memory_usage = (torch.cuda.memory_allocated(device) + torch.cuda.memory_reserved(device))/1024**3
        if memory_usage > max_gpu_usage_allowed:
            if logger is not None:
                logger.info('All GPUs are busy, please wait for idle GPU or cancel the automatic GPU allocation.')
            raise SystemExit('No GPU is idle.')
        if device.index != config['gpu_id']:
            if logger is not None:
                logger.info(f'Change to device cuda:{device.index}.')
        config['device'] = device

def create_hyper_result_path(output_path, model_name):
    r"""Create file path for the result of hyper-parameters. 

        There is an incrementing id at the end of the filename to avoid duplicate filenames.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    res_count = 0
    file_path = os.path.join(output_path, '{}{:d}.result'.format(model_name, res_count))
    while os.path.exists(file_path):
        res_count += 1
        file_path = os.path.join(output_path, '{}{:d}.result'.format(model_name, res_count))
    return file_path

def create_log_name(log_name, model_name):
    r"""Create file name for ``logger``. 

        There is an incrementing id at the end of the filename to avoid duplicate filenames.
    """
    log_count = 0
    file_path = os.path.join(model_name, '{}-{:d}.log'.format(log_name, log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(model_name, '{}-{:d}.log'.format(log_name, log_count))
    return '{}-{:d}'.format(log_name, log_count)

log_colors_config = {
    'DEBUG': 'cyan',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}

class RemoveColorFilter(logging.Filter):

    def filter(self, record):
        if record:
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            record.msg = ansi_escape.sub('', str(record.msg))
        return True


def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'


def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Example:
        >>> logger = logging.getLogger(config)
        >>> logger.debug(train_state)
        >>> logger.info(train_result)
    """
    init(autoreset=True)
    LOGROOT = './log/'
    dir_name = os.path.dirname(LOGROOT)
    ensure_dir(dir_name)
    model_name = os.path.join(dir_name, config['model'])
    ensure_dir(model_name)
    if config['log_name'] is not None:
        logfilename = '{}/{}.log'.format(config['model'], create_log_name(config['log_name'], model_name))
    else:
        logfilename = '{}/{}.log'.format(config['model'], get_local_time())

    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(log_color)s%(asctime)-15s %(levelname)s  %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = colorlog.ColoredFormatter(sfmt, sdatefmt, log_colors=log_colors_config)
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)
    remove_color_filter = RemoveColorFilter()
    fh.addFilter(remove_color_filter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(level=level, handlers=[sh, fh])