import numpy as np

try:
    import numba
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False

INT = np.int32

VERBOSE_MODE = True


def is_verbose():
    return VERBOSE_MODE


def set_verbose(verbose_mode):
    global VERBOSE_MODE
    VERBOSE_MODE = verbose_mode


def _speed_up(func):
    """A conditional decorator that use numba to speed up the function"""
    if USE_NUMBA:
        import numba
        return numba.njit(func)
    else:
        return func
