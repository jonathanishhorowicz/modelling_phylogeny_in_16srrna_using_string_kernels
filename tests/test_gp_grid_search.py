import pytest
import sys
from pathlib import Path

import gpflow as gpf
from gpflow.kernels import Linear
from sklearn.datasets import load_diabetes
import numpy as np

SCRIPTS_PATH = str((Path(__file__).parent.parent / 'scripts').resolve())
if SCRIPTS_PATH not in sys.path:
    sys.path.append(str(SCRIPTS_PATH))
from gp_utils import hparam_grid_search

# @pytest.fixture
# def scripts_path():
#     return (Path(__file__).parent.parent / 'scripts').resolve()

# @pytest.fixture(autouse=True)
# def add_scripts_to_path(scripts_path):
#     pytest.set_trace()
#     sys.path.append(str(scripts_path))
#     yield
#     sys.path.remove(str(scripts_path))

# @pytest.fixture(autouse=True)
# def import_gp_utils(add_scripts_to_path):
#     from gp_utils import hparam_grid_search

@pytest.fixture
def dummy_regression_dataset():
    X, y = load_diabetes(return_X_y=True)
    y -= y.mean()
    y /= y.var()
    return X, y[:,np.newaxis]

@pytest.fixture
def param_values():
    return {
        'signal_variance': np.logspace(-5, -1, 10),
        'noise_variance': np.logspace(-5, -1, 10)
    }

def fit_function_factory(**kwargs):

    signal_variance = kwargs['signal_variance']
    noise_variance = kwargs['noise_variance']

    def _fit_fn(X, y):
        return gpf.models.GPR(
            data=(X, y),
            kernel=Linear(variance=signal_variance),
            noise_variance=noise_variance
        )
    
    return _fit_fn




def test_hparam_grid_search(dummy_regression_dataset, param_values):
    hparam_grid_search(dummy_regression_dataset, fit_function_factory, param_values)