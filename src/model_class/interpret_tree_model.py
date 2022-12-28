# Author: Caesar Wong
# Date: 2022-12-28

"""
A script that defines the model interpreter class & methods for the tree-based model
Tree explainer in the SHAP library is used
"""

import shap
from sklearn.pipeline import Pipeline

class tree_model_interpreter:
    """
    Model Interpreter with the SHAP plotting capabilities

    Attributes
    ----------
    model : sklearn model
        main model for interpretation
    X_train : df
        X (features) of the training data 
    y_train: df
        y (target) of the training data
    X_test : df
        X (features) of the testing data 
    y_test: df
        y (target) of the testing data
    out_dir : str
        output directory path

    Methods
    -------
    __init__(model, X_train, y_train, X_test, y_test, out_dir)
        object initialization function
    get_tree_explainer()
        obtain the SHAP tree explainer object
    get_shap_values_on_train()
        return shap values on training data
    get_shap_values_on_test()
        return shap values on testing data
    get_dependence_plot(feat_1)
        return the dependence plot with a given feature
    get_summary_plot()
        return shap summary plot
    get_force_plot(sample_ind, shap_values, dataset)
        return shap force plot given the data row index, shap values and dataset
    """
    def __init__(self, model, X_train, y_train, X_test, y_test, out_dir):
        self.model = model # tree based model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.out_dir = out_dir # model output directory

    
    def get_tree_explainer(self):
        '''
        obtain the SHAP tree explainer object
        
        Parameters
        ----------

        Returns
        -------
        tree explainer object
        '''
        if (isinstance(self.model, Pipeline)):
            return shap.TreeExplainer(self.model[-1])
        else:
            return shap.TreeExplainer(self.model)

    def get_shap_values_on_train(self):
        '''
        return shap values on training data
        
        Parameters
        ----------
        
        Returns
        -------
        shap values : df
        '''
        return self.get_tree_explainer().shap_values(self.X_train)

    def get_shap_values_on_test(self):
        '''
        return shap values on testing data
        
        Parameters
        ----------
        
        Returns
        -------
        shap values : df
        '''
        return self.get_tree_explainer().shap_values(self.X_test)

    def get_dependence_plot(self, feat_1):
        '''
        return the dependence plot with a given feature
        
        Parameters
        ----------
        feat_1 : str
            feature name for plotting the dependence plot
        Returns
        -------
        shap dependence plot
        '''
        return shap.dependence_plot(feat_1, self.get_shap_values_on_train()[0], self.X_train, show=False)

    def get_summary_plot(self):
        '''
        return the summary plot
        
        Parameters
        ----------

        Returns
        -------
        shap summary plot
        '''
        return shap.summary_plot(self.get_shap_values_on_train()[0], self.X_train, show=False)

    def get_force_plot(self, sample_ind, shap_values, dataset):
        '''
        return the force plot with the data row index, shap values and dataset
        
        Parameters
        ----------
        sample_ind : int
            index of the data row for exploring the force plot
        shap_values : df 
            shap values from SHAP
        dataset : df
            data we are interested to investigate

        Returns
        -------
        shap force plot
        '''
        return shap.force_plot(
                    self.get_tree_explainer().expected_value[0],
                    shap_values[0][sample_ind, :],
                    dataset.iloc[sample_ind, :],
                    matplotlib=True,
                    text_rotation=15,
                    show=False
                )