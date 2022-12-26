# Author: Caesar Wong
# Date: 2022-12-24

"""
Configuration script for yacs library, mainly for storing default model configuration value


Usage: 

"""

from yacs.config import CfgNode as CN
from scipy.stats import lognorm, loguniform, randint
from sklearn.metrics import f1_score, make_scorer, recall_score

_C = CN()

_C.MODEL = CN()
# Model's random state
_C.MODEL.RANDOM_STATE = 123
# Model's strategy (dummy)
_C.MODEL.STRATEGY = "stratified"
# Model's max iteration
_C.MODEL.MAX_ITER = 1000
# Model's class weight
_C.MODEL.CLASS_WEIGHT = "balanced"

_C.MODEL.N_ITER=50
_C.MODEL.VERBOSE=1
_C.MODEL.N_JOBS=1
_C.MODEL.SCORING=[make_scorer(f1_score, average="macro")]
_C.MODEL.RETURN_TRAIN_SCORE=True


# Linear Model related config
_C.LINEAR_MODEL = CN()
# Linear Model's parameter grid (for CV search)
_C.LINEAR_MODEL.PARAM_GRID = [{"logisticregression__C": loguniform(1e-3, 1e3)}]

# Tree-based model hyperparameter (param_grid)
_C.TREE_MODEL = CN()
# list of params_grid
_C.TREE_MODEL.PARAM_GRID = [
{
    "randomforestclassifier__n_estimators": randint(low=10, high=100),
    "randomforestclassifier__max_depth": randint(low=2, high=20),
},
{
    "xgbclassifier__n_estimators": randint(10, 100),
    "xgbclassifier__max_depth": randint(low=2, high=20),
    "xgbclassifier__learning_rate": [0.01, 0.1],
    "xgbclassifier__subsample": [0.5, 0.75, 1],
},
{
    "lgbmclassifier__n_estimators": randint(10, 100),
    "lgbmclassifier__max_depth": randint(low=2, high=20),
    "lgbmclassifier__learning_rate": [0.01, 0.1],
    "lgbmclassifier__subsample": [0.5, 0.75, 1],
}
]



def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for this project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
