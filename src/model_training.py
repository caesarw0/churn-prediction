# Author: Caesar Wong
# Date: 2022-12-24

"""
A script that trains different model & store the CV result

Usage: src/model_training.py --train=<train> --scoring_metric=<scoring_metric> --out_dir=<out_dir> --target_name=<target_name>
 
Options:
--train=<train>                     Input path for the train dataset
--target_name=<target_name>         Target name for supervised learning
--scoring_metric=<scoring_metric> Scoring Metrics for classification (e.g. 'f1', 'recall', 'precision')
--out_dir=<out_dir>                 Path to directory where the serialized model should be written

"""
# Example:
# python model_training.py --train="../data/processed/train.csv" --scoring_metric="f1" --target_name="Exited" --out_dir="../results/"


# import
from docopt import docopt
import os
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

from config.model_config import get_cfg_defaults     # import yacs config method (with default config written in the script)

from utils.get_logger import return_logger

logger = return_logger()

opt = docopt(__doc__) # This would parse into dictionary in python


from model_class.basicmodel import basicmodel

def main(train, scoring_metric, target_name, out_dir):

    # get DEFAULT model configuration
    cfg = get_cfg_defaults()
    cfg.freeze()

    if scoring_metric == "macro":
        custom_scorer = make_scorer(f1_score, average="macro")
        scoring_metric = custom_scorer

    logger.info("Begin Model Training")
    cross_val_results = {}
    # get training data
    train_df = pd.read_csv(train)
    X_train, y_train = train_df.drop(columns=[target_name]), train_df[target_name]
    
    # dummy
    logger.info("Dummy model training")
    dummy_model = basicmodel("dummy", DummyClassifier(strategy=cfg.MODEL.STRATEGY), out_dir)
    dummy_model.model.fit(X_train, y_train)
    cross_val_results["dummy"] = dummy_model.mean_std_cross_val_scores(X_train, y_train, scoring=scoring_metric)
    dummy_model.model_saving()
    logger.info("\n" + str(cross_val_results["dummy"]))

    # linear
    logger.info("Logistic Regression model training")
    log_model = basicmodel("logreg", LogisticRegression(max_iter=cfg.MODEL.MAX_ITER, class_weight=cfg.MODEL.CLASS_WEIGHT), out_dir)
    log_model.model.fit(X_train, y_train)
    cross_val_results["logreg"] = log_model.mean_std_cross_val_scores(X_train, y_train, scoring=scoring_metric)
    log_model.model_saving()
    logger.info("\n" + str(cross_val_results["logreg"]))

    # linear + tuning

    

    # advance model

    # advance model + feat_sel

    # advance model + tuning

    pass

if __name__ == "__main__":
    main(opt["--train"], opt["--scoring_metric"], opt["--target_name"], opt["--out_dir"])