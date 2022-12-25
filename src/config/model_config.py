# Author: Caesar Wong
# Date: 2022-12-24

"""
Configuration script for yacs library, mainly for storing default model configuration value


Usage: 

"""

from yacs.config import CfgNode as CN
from scipy.stats import lognorm, loguniform, randint


_C = CN()

_C.MODEL = CN()
# Model's random state
_C.MODEL.RANDOM_STATE = 2
# Model's strategy (dummy)
_C.MODEL.STRATEGY = "stratified"
# Model's max iteration
_C.MODEL.MAX_ITER = 1000
# Model's class weight
_C.MODEL.CLASS_WEIGHT = "balanced"

# Linear Model related config
_C.LINEAR_MODEL = CN()
# Linear Model's parameter grid (for CV search)
_C.LINEAR_MODEL.PARAM_GRID = [{"logisticregression__C": loguniform(1e-3, 1e3)}]



def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for this project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
