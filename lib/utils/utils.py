import datetime
import logging
import logging.handlers
import os
import pickle
import tempfile
import time

from contextlib import contextmanager
from functools import wraps


REAL_PATH = os.path.dirname(os.path.realpath(__file__))

dd=breakpoint


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _get_logger_path():
    KST = datetime.timezone(datetime.timedelta(hours=9))
    # now = datetime.datetime.now(tz=KST).strftime('%Y%m%d-%H%M%S')
    now = datetime.datetime.now(tz=KST).strftime('%Y%m%d')
    
    logger_path = os.path.join(REAL_PATH, 'result', 'logs')
    check_path(logger_path)

    filename = now + '.log'
    filepath = os.path.join(logger_path, filename)
    return filepath


logger = logging.getLogger(__name__)

stream_formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
file_formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    
stream_handler = logging.StreamHandler()
logger_path = _get_logger_path()
file_handler = logging.FileHandler(logger_path)

stream_handler.setFormatter(stream_formatter)
file_handler.setFormatter(file_formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

logger.setLevel(level=logging.DEBUG)


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time() - t1
        logger.info('Running time [ {} ]: {:.3f} sec'.format(func.__name__, t2))
        return result
    return wrapper


@contextmanager
def _tempfile(*args, **kwargs):
    fd, name = tempfile.mkstemp(*args, **kwargs)
    os.close(fd)
    try:
        yield name
    finally:
        try:
            os.remove(name)
        except OSError as e:
            if e.errno == 2:
                pass
            else:
                raise e


@contextmanager
def _open_atomic(filepath, *args, **kwargs):
    fsync = kwargs.pop('fsync', False)
    
    with _tempfile(dir=os.path.dirname(filepath)) as tmppath:
        with open(tmppath, *args, **kwargs) as f:
            yield f
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.rename(tmppath, filepath)


def safe_pickle_dump(obj, fname):
    with _open_atomic(fname, 'wb') as f:
        pickle.dump(obj, f, -1)


def pickle_load(fname):
    try:
        db = pickle.load(open(fname, 'rb'))
    except Exception as e:
        logger.error('error loading existing database:\n{}\nstarting from an empty database'.format(e))
        db = {}
    return db




class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count