# Author: Caesar Wong
# Date: 2022-12-24

"""
A script that trains different model & store the CV result

Usage: src/model_training.py --train=<train> --out_dir=<out_dir> --target_name=<target_name>
 
Options:
--train=<train>                     Input path for the train dataset
--target_name=<target_name>         Target name for supervised learning
--scoring_metric=<scoring_metric> Scoring Metrics for classification (e.g. 'f1', 'recall', 'precision')
--out_dir=<out_dir>                 Path to directory where the serialized model should be written

"""
# Example:
# python model_training.py --train="../data/processed/train.csv" --target_name="Exited" --out_dir="../results/"


# import
from docopt import docopt
import os
import numpy as np
import pandas as pd
from scipy.stats import lognorm, loguniform, randint
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, recall_score
from sklearn.dummy import DummyClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel

from config.model_config import get_cfg_defaults     # import yacs config method (with default config written in the script)

from utils.get_logger import return_logger

logger = return_logger()

opt = docopt(__doc__) # This would parse into dictionary in python

# get DEFAULT model configuration
cfg = get_cfg_defaults()
cfg.freeze()

from model_class.basicModel import basicModel

def define_model(X_train, y_train, out_dir):

    # sequence of model definition
    model_list = []
    log_model = LogisticRegression(max_iter=cfg.MODEL.MAX_ITER, class_weight=cfg.MODEL.CLASS_WEIGHT)
    ratio = np.bincount(y_train)[0] / np.bincount(y_train)[1]
    feat_sel = SelectFromModel(
                LogisticRegression(solver="liblinear", penalty="l1", max_iter=1000)
            )

    model_dict = {"dummy": DummyClassifier(strategy=cfg.MODEL.STRATEGY),
                "logreg": log_model,
                "logreg (tuned)": RandomizedSearchCV(
                                    make_pipeline(
                                        log_model
                                    ),
                                    cfg.LINEAR_MODEL.PARAM_GRID[0],
                                    n_iter=cfg.MODEL.N_ITER,
                                    verbose=cfg.MODEL.VERBOSE,
                                    scoring=cfg.MODEL.SCORING[0],
                                    random_state=cfg.MODEL.RANDOM_STATE,
                                    return_train_score=cfg.MODEL.RETURN_TRAIN_SCORE,
                                ),
                "random forest": RandomForestClassifier(class_weight=cfg.MODEL.CLASS_WEIGHT, random_state=cfg.MODEL.RANDOM_STATE),
                "xgboost": XGBClassifier(scale_pos_weight=ratio, random_state=cfg.MODEL.RANDOM_STATE),
                "lgbm": LGBMClassifier(scale_pos_weight=ratio, random_state=cfg.MODEL.RANDOM_STATE),
                }
    for adv_key in list(model_dict.keys())[-3:]:
        model_dict[adv_key + ' + feat_sel'] = make_pipeline(feat_sel, model_dict[adv_key])
    count = 0
    for adv_key in list(model_dict.keys())[-6:-3]:
        model_dict[adv_key + ' (tuned)'] = RandomizedSearchCV(
                                                model_dict[adv_key] ,
                                                cfg.TREE_MODEL.PARAM_GRID[count],
                                                n_iter=cfg.MODEL.N_ITER,
                                                verbose=cfg.MODEL.VERBOSE,
                                                scoring=cfg.MODEL.SCORING[0],
                                                random_state=cfg.MODEL.RANDOM_STATE,
                                                return_train_score=cfg.MODEL.RETURN_TRAIN_SCORE
                                            )

        count+=1

    for model_name, model in model_dict.items():
        logger.info("creating model: " + model_name)
        model_list.append(basicModel(model_name, model, X_train, y_train, cfg.MODEL.SCORING[0], out_dir))
    
    return model_list

def model_train_save(model_list):

    for model in model_list:
        model.model_training_saving()
    pass

def get_cv_result(model_list):
    
    cross_val_results = {}

    for model in model_list:
        cross_val_results[model.name] = model.mean_std_cross_val_scores(scoring=cfg.MODEL.SCORING[0])

    return cross_val_results


def main(train, target_name, out_dir):

    logger.info("Begin Model Training")
    cross_val_results = {}
    # get training data
    train_df = pd.read_csv(train)
    X_train, y_train = train_df.drop(columns=[target_name]), train_df[target_name]
    
  
    define_model(X_train, y_train, out_dir)

    
    


if __name__ == "__main__":
    main(opt["--train"], opt["--target_name"], opt["--out_dir"])