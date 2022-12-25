# Author: Caesar Wong
# Date: 2022-12-24

"""
A python class that defines the basic modelling (e.g. dummy, regressor, classifier)

"""

import pandas as pd
from sklearn.model_selection import cross_validate
import os
import pickle

class basicmodel:

    def __init__(self, name, model, out_dir):
        self.name = name # model name
        self.model = model # sklearn / other model object / pipeline / best estimator
        self.out_dir = out_dir # model output directory

    def mean_std_cross_val_scores(self, X_train, y_train, **kwargs):
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

        scores = cross_validate(self.model, X_train, y_train, return_train_score=True,**kwargs)

        mean_scores = pd.DataFrame(scores).mean()
        std_scores = pd.DataFrame(scores).std()
        out_col = []

        for i in range(len(mean_scores)):
            out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

        return pd.Series(data=out_col, index=mean_scores.index)


    def model_saving(self):

        try:
            file_log = open(self.out_dir + '/model_' + self.name, 'wb')
        except:
            os.makedirs(os.path.dirname(self.out_dir))
            file_log = open(self.out_dir + '/model_' + self.name, 'wb')

        pickle.dump(self.model, file_log)