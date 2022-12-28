

import matplotlib.pyplot as plt
import eli5
import shap
from sklearn.pipeline import Pipeline
# shap value

# shap plot
    # shap dependence_plot
    # shap summary_plot
    # shap force_plot


class tree_model_interpreter:

    def __init__(self, name, model, X_train, y_train, X_test, y_test, out_dir):
        self.name = name # model name
        self.model = model # tree based model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.out_dir = out_dir # model output directory

    
    def get_tree_explainer(self):
        if (isinstance(self.model, Pipeline)):
            return shap.TreeExplainer(self.model[-1])
        else:
            return shap.TreeExplainer(self.model)

    def get_shap_values_on_train(self):
        return self.get_tree_explainer().shap_values(self.X_train)

    def get_shap_values_on_test(self):
        return self.get_tree_explainer().shap_values(self.X_test)

    def get_dependence_plot(self, feat_1):
        return shap.dependence_plot(feat_1, self.get_shap_values_on_train()[0], self.X_train, show=False)

    def get_summary_plot(self):
        return shap.summary_plot(self.get_shap_values_on_train()[0], self.X_train, show=False)

    def get_force_plot(self, sample_ind, shap_values, dataset):
        return shap.force_plot(
                    self.get_tree_explainer().expected_value[0],
                    shap_values[0][sample_ind, :],
                    dataset.iloc[sample_ind, :],
                    matplotlib=True,
                    text_rotation=15,
                    show=False
                )