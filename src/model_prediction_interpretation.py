# Author: Caesar Wong
# Date: 2022-12-28

"""
A script that test and plot the result for the selected best model (defined in the config file) 
and store it under the results/prediction_output folder.

Usage: src/model_prediction_interpretation.py --train=<train> --test=<test> --target_name=<target_name> --target_names=<target_names> --model_path=<model_path> --out_dir=<out_dir>
 
Options:
--train=<train>               Input path for the train dataset
--test=<test>               Input path for the test dataset
--target_name=<target_name> Target name for supervised learning
--target_names=<target_names> Target names for supervised learning, e.g. "class0,class1"
--model_path=<model_path>   Model path storing all the pre-trained model
--out_dir=<out_dir>         Path to directory where the serialized model should be written

"""

# Example:
# python model_prediction.py --train="../data/processed/train.csv" --test="../data/processed/test.csv" --target_name="Exited" --target_names="Stayed,Exited" --model_path="../results/model_output/" --out_dir="../results/prediction_output/"

# import
from docopt import docopt
import pickle
import pandas as pd
import numpy as np


from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, plot_confusion_matrix

import matplotlib.pyplot as plt

from config.model_config import get_cfg_defaults     # import yacs config method (with default config written in the script)

from model_class.interpret_tree_model import tree_model_interpreter

from utils.get_logger import return_logger

logger = return_logger()

opt = docopt(__doc__) # This would parse into dictionary in python

# get DEFAULT model configuration
cfg = get_cfg_defaults()
cfg.freeze()


def get_model(model_path):
    '''
    obtain the sklearn model given the model path & model name (defined in the config file)
    
    Parameters
    ----------
    model_path : str
        model path storing the desired model

    Returns
    -------
    sklearn model
    '''
    model_name = cfg.MODEL_INTERPRET.NAME
    cur_file = open(model_path + 'model_' + model_name, 'rb')
    return pickle.load(cur_file)

def generate_cf_matrix_plot(model, X_data, y_data, target_names, out_dir):
    '''
    generate classification report & save it to out_dir
    
    Parameters
    ----------
    model : sklearn model
        supervised learning model with defined parameters
    X_data : df
        X (features) of the given data
    y_data : df
        y (target) of the given data
    target_names : str 
        with more than 1 class name separated by ',' (e.g. "Stayed,Exited" / "Class0,Class1")
    out_dir : str
        output directory path

    Returns
    -------
    save the confusion matrix plot
    '''
    plot_confusion_matrix(
        model,
        X_data,
        y_data,
        display_labels=target_names,
        values_format="d",
        cmap=plt.cm.Blues,
    )
    plt.savefig(out_dir  + 'confusion_matrix_test.png', dpi=200, bbox_inches='tight')
    plt.clf()

def generate_classification_report(model, X_data, y_data, target_names, out_dir):
    '''
    generate classification report & save it to out_dir
    
    Parameters
    ----------
    model : sklearn model
        supervised learning model with defined parameters
    X_data : df
        X (features) of the given data
    y_data : df
        y (target) of the given data
    target_names : str 
        with more than 1 class name separated by ',' (e.g. "Stayed,Exited" / "Class0,Class1")
    out_dir : str
        output directory path

    Returns
    -------
    save the classification report
    '''
    predictions = model.predict(X_data)
    report = classification_report(
            y_data, predictions, target_names=target_names, output_dict=True
        )
    df_classification_report = pd.DataFrame(report).transpose()
    logger.info(df_classification_report)
    df_classification_report.to_csv(out_dir + '/classification_report.csv')


def generate_force_plot_test(interpreter, X_test, y_test):
    '''
    generate force plot & save it in the out_dir
    
    Parameters
    ----------
    interpreter : tree_model_interpreter object
        tree based model interpreter
    X_test : df
        X (features) of the testing data 
    y_test: df
        y (target) of the testing data

    Returns
    -------
    save the force plot
    '''
    # shap_value_on_train = interpreter.get_shap_values_on_train()
    shap_value_on_test = interpreter.get_shap_values_on_test()
    y_test_reset = y_test.reset_index(drop=True)
    class_0_ind = y_test_reset[y_test_reset == 0].index.tolist()
    class_1_ind = y_test_reset[y_test_reset == 1].index.tolist()

    class_0_index = class_0_ind[0]
    class_1_index = class_1_ind[1]

    force_plot_1 = interpreter.get_force_plot(class_0_index, shap_value_on_test, X_test)
    plt.savefig(interpreter.out_dir + 'force_plot_1_test.png',dpi=200,bbox_inches='tight')
    plt.clf()

    force_plot_2 = interpreter.get_force_plot(class_1_index, shap_value_on_test, X_test)
    plt.savefig(interpreter.out_dir + 'force_plot_2_test.png',dpi=200,bbox_inches='tight')
    plt.clf()

def generate_summary_plot(interpreter):
    '''
    generate summary plot & save it in the out_dir
    
    Parameters
    ----------
    interpreter : tree_model_interpreter object
        tree based model interpreter
     
    Returns
    -------
    save the summary plot
    '''
    interpreter.get_summary_plot()
    plt.savefig(interpreter.out_dir + 'summary_plot.png',dpi=200,bbox_inches='tight')
    plt.clf()

def generate_dependence_plot(interpreter, feat_name):
    '''
    generate dependence plot & save it in the out_dir
    
    Parameters
    ----------
    interpreter : tree_model_interpreter object
        tree based model interpreter
    feat_name : str
        selected feature name
     
    Returns
    -------
    save the dependence plot
    '''
    interpreter.get_dependence_plot(feat_name)
    plt.savefig(interpreter.out_dir + 'dependence_plot.png',dpi=200,bbox_inches='tight')
    plt.clf()



def main(train, test, target_name, target_names, model_path, out_dir):
    '''
    read model from model path (model name is specified in the config file),
    generate confusion matrix, classification report, SHAP plot (dependence plot, summary plot, force plot)
    
    Parameters
    ----------
    train : str
        training data path
    test : str
        testing data path
    target_name : str
        class label name (e.g. "Exited" / "Target")
    target_names : str 
        with more than 1 class name separated by ',' (e.g. "Stayed,Exited" / "Class0,Class1")
    model_path : str
        path storing all the necessary model
    out_dir : str
        output directory path
     
    Returns
    -------
    Output the following items in the results/prediction_output/ folder
    - confusion matrix png
    - dependence plot png
    - force plot png (2)
    - summary plot png
    - classification report csv
    Examples
    --------
    >>> main(opt["--train"], opt["--test"], opt["--target_name"], opt["--target_names"], opt["--model_path"], opt["--out_dir"])
    '''
    logger.info("Begin Model Prediction & Interpretation ...")
    # get train data
    train_df = pd.read_csv(train, float_precision='round_trip')
    X_train, y_train = train_df.drop(columns=[target_name]), train_df[target_name]

    # get test data
    test_df = pd.read_csv(test, float_precision='round_trip')
    X_test, y_test = test_df.drop(columns=[target_name]), test_df[target_name]

    # print(y_test.value_counts())
    selected_model = get_model(model_path)
    
    target_name_list = target_names.split(",")

    tree_interpreter = tree_model_interpreter(
                        selected_model, X_train, y_train, X_test, y_test, out_dir
                        )
    # for testing data
    generate_force_plot_test(tree_interpreter, X_test, y_test)

    generate_cf_matrix_plot(selected_model, X_test, y_test, target_name_list, out_dir)

    generate_classification_report(selected_model, X_test, y_test, target_name_list, out_dir)

    generate_summary_plot(tree_interpreter)

    values = np.abs(tree_interpreter.get_shap_values_on_train()[0]).mean(0)
    shap_with_feat_name = pd.DataFrame(data=values, index=X_train.columns, columns=["SHAP"]).sort_values(
        by="SHAP", ascending=False
    )[:10]
    
    generate_dependence_plot(tree_interpreter, shap_with_feat_name.index[0])
    
    logger.info("Complete Model Prediction & Interpretation")


if __name__ == "__main__":
    main(opt["--train"], opt["--test"], opt["--target_name"], opt["--target_names"], opt["--model_path"], opt["--out_dir"])