from typing import Callable, Tuple, Dict
from numpy.typing import ArrayLike
from sklearn.model_selection import ParameterGrid
import numpy as np
import pandas as pd

import logging
logger = logging.getLogger('__name__')

# def _is_montonic(vals: ArrayLike) -> bool:
#     vals_as_series = pd.Series(vals)
#     return vals_as_series.is_monotonic_decreasing or vals_as_series.is_monotonic_increasing

def hparam_grid_search(
        data: Tuple[ArrayLike, ArrayLike],
        fit_fn_factory: Callable,
        param_values: Dict[str, ArrayLike]
    ):
    X, y = data
    logger.info('Running grid search on data with {:,} samples'.format(X.shape[0]))
    param_grid = ParameterGrid(param_values)
    logger.info('Parameter grid has len {:,}'.format(len(param_grid)))
    d_param_ranges = {
        param_name: (vals.min(), vals.max()) for param_name, vals in param_values.items()
    }
    logger.info('Param ranges: {}'.format(d_param_ranges))

    logger.info('Running grid search...')
    obj_fun_vals = np.empty(len(param_grid), dtype=np.float64)
    for idx, param_dict in enumerate(param_grid):
        logger.debug('Fitting model {} of {}'.format(idx, len(param_grid)))
        fit_fn = fit_fn_factory(**param_dict)
        m = fit_fn(X, y)
        obj_fun_vals[idx] = m.log_posterior_density().numpy()

    
    best_idx = np.nanargmax(obj_fun_vals)
    best_params = list(param_grid)[best_idx]
    logger.info('Best params are: {} (grid position {})'.format(best_params, best_idx))

    # print warning if any of the optimal parameter values are at the ends of its range
    for param, range in d_param_ranges.items():
        if np.isclose(best_params[param], range[0]) or np.isclose(best_params[param], range[1]):
            logger.warning('Param "{}" is at the limit of its grid range'.format(param))

    best_m = fit_fn_factory(**best_params)(X, y)
    return best_m