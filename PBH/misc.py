import numpy as np
from scipy.interpolate import interp1d

def log_interp1d(x, y, which = 'both', kind='linear'):
    """
    Returns interpolator function where y is in log-scale
    """

    if which == 'both':
        log_x = np.log10(x)
        log_y = np.log10(y)

    elif which == 'x':
        log_x = np.log10(x)
        log_y = y

    elif which == 'y':
        log_x = x
        log_y = np.log10(y)

    else:
        raise(ValueError("which must be 'both', 'x' or 'y', refering to the variable given in logscale."))

    lin_int = interp1d(log_x, log_y, kind = kind, fill_value="extrapolate")

    log_int = lambda z: np.power(10.0, lin_int(z))
    
    return log_int

