# Author: Caesar Wong
# Date: 2022-12-25

"""
A python class that defines the basic modelling (e.g. dummy, regressor, classifier)

"""

import pandas as pd
from sklearn.model_selection import cross_validate

import os
import pickle

from utils.get_logger import return_logger
logger = return_logger()


class basicModel:

    def __init__(self, name, model, X_train, y_train, scoring_metric, out_dir):
        self.name = name # model name
        self.model = model # sklearn / other model object / pipeline / best estimator
        self.X_train = X_train
        self.y_train = y_train
        self.scoring_metric = scoring_metric
        self.out_dir = out_dir # model output directory

    def mean_std_cross_val_scores(self, **kwargs):
        """
        Returns mean and std of cross validation

        Parameters
        ----------
        model :
            scikit-learn model
        X_train : numpy array or pandas DataFrame
            X in the training data
        y_train :
            y in the training data

        Returns
        ----------
            pandas Series with mean scores from cross_validation

        Examples
        --------
        >>> model.mean_std_cross_val_scores(X_train, y_train)
        fit_time: xxxx (+/- xxxx)
        score_time: xxxx (+/- xxxx)
        train_score: xxxx (+/- xxxx)
        test_score: xxxx (+/- xxxx)
        """
        logger.info(self.name + " model cross validation ...")

        scores = cross_validate(self.model, self.X_train, self.y_train, return_train_score=True,**kwargs)

        mean_scores = pd.DataFrame(scores).mean()
        std_scores = pd.DataFrame(scores).std()
        out_col = []

        for i in range(len(mean_scores)):
            out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

        return pd.Series(data=out_col, index=mean_scores.index)


    def model_saving(self):
        logger.info(self.name + " model saving ...")
        try:
            file_log = open(self.out_dir + '/model_' + self.name, 'wb')
        except:
            os.makedirs(os.path.dirname(self.out_dir))
            file_log = open(self.out_dir + '/model_' + self.name, 'wb')

        pickle.dump(self.model, file_log)

    def model_training_saving(self):

       
        logger.info(self.name + " model training ...")
        self.model.fit(self.X_train, self.y_train)
        self.model_saving()
        logger.info(self.name + " model training & saving completed")

    