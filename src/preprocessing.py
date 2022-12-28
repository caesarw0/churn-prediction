# Author: Caesar Wong
# Date: 2022-12-23

"""
A script that preprocess the data from data/raw/ folder and store it in the data/preprocess/ folder.

Usage: src/preprocessing.py --input_path=<input_path> --sep=<sep> --test_size=<test_size> --random_state=<random_state> --output_path=<output_path> 

Options:
--input_path=<input_path>       Input path for the raw dataset
--sep=<sep>                     Separator for reading the CSV file
--test_size=<test_size>         Test size for train_test_split
--random_state=<random_state>   Random state for train_test_split
--output_path=<output_path>     Specify the path where user can store the preprocessed dataframe

"""

# Example:
# python preprocessing.py --input_path="../data/raw/data.csv" --sep=',' --test_size=0.3 --random_state=123 --output_path="../data/processed"

# importing necessary modules
from docopt import docopt

from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.compose import  make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.get_logger import return_logger

logger = return_logger()

opt = docopt(__doc__) # This would parse into dictionary in python

def generalPreprocessing(df):
    '''
    Perform general preprocessing on df
    
    Parameters
    ----------
    df : pd.DataFrame
        dataframe storing all the raw data
    
    Returns
    -------
    df : pd.DataFrame
        preprocessed (fix type, drop 'Enrolled', arrange column order) dataframe
        
    Examples
    --------
    >>> generalPreprocessing(df)
    <df object>
    '''
    
    # testing column shape
    try:
        assert df.shape[1] == 14
        logger.info("Correct dataframe shape")
    except AssertionError:
        logger.error("Wrong dataframe shape (incorrect number of columns)123")
        raise

    # drop na from df
    df = df.dropna()

    # Dropping unnecessary columns
    df.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace=True)
    
    try:
        assert df.shape[1] == 11
        logger.info("Correct dataframe shape after dropping unnecessary columns")
    except:
        logger.error("Wrong dataframe shape after dropping unnecessary columns")
        raise

    logger.info("df shape : " + str(df.shape))
    return df

def columnTransformation(train_df, test_df):
    '''
    Perform column transformation (OHE, standard scaling, binary variable encoding) on the given data
    
    Parameters
    ----------
    train_df : pd.DataFrame
        training data

    test_df : pd.DataFrame
        testing data
    
    Returns
    -------
    transformed_train_df : pd.DataFrame
        column transformed training data

    transformed_test_df : pd.DataFrame
        column transformed testing data

    Examples
    --------
    >>> transformed_train_df, transformed_test_df = columnTransformation(train_df, test_df)
    
    '''
    # perform column transformation
    # - binary
    # - categorical (OHE)
    # - continuous (standardscaler)
    binary_features = ['Gender', 'HasCrCard', 'IsActiveMember']
    categorical_features = ['Geography']
    numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary']
    target_column = ['Exited']

    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
        (OneHotEncoder(drop="if_binary", dtype=int), binary_features),
        ("passthrough", target_column)
    )

    # fit preprocessor & transform training data
    train_data = preprocessor.fit_transform(train_df)
    categorical_feats = preprocessor.named_transformers_['onehotencoder-1'].get_feature_names_out(categorical_features).tolist()
    binary_feats = preprocessor.named_transformers_['onehotencoder-2'].get_feature_names_out(binary_features).tolist()

    feature_names = numeric_features + categorical_feats + binary_feats + target_column
    transformed_train_df = pd.DataFrame(train_data, columns=feature_names)

    # transform testing data
    test_data = preprocessor.transform(test_df)
    transformed_test_df = pd.DataFrame(test_data, columns=feature_names)

    return transformed_train_df, transformed_test_df


def main(input_path, sep, test_size, random_state, output_path):
    '''
    main function for the preprocessing script
    1. Data dropping
    2. Data arrange column
    3. column transformation

    Parameters
    ----------
    input_path : str
        input file path

    sep : str
        separator for read_csv

    test_size : str --> float
        testing size for train_test_split

    random_state : str --> int
        random state for read_csv

    output_path : str
        output file path
    
    Returns
    -------
    <None>
    save 2 csv files to output_path
    - train.csv
    - test.csv

    Examples
    --------
    >>> main(opt["--input_path"], opt["--sep"], opt["--test_size"], opt["--random_state"], opt["--output_path"])
    
    '''
    df = pd.read_csv(input_path, sep=sep)
    logger.info("Begin General Preprocessing")
    df = generalPreprocessing(df)
    logger.info("Finish General Preprocessing")
    # data splitting
    train_df, test_df = train_test_split(df, test_size=float(test_size), random_state=int(random_state))

    logger.info("train_df shape : " + str(train_df.shape))

    logger.info("Storing Data")

    # calling local columnTransformation function
    transformed_train_df, transformed_test_df = columnTransformation(train_df, test_df)

    transformed_train_df.to_csv(output_path + '/train.csv', index=False)
    transformed_test_df.to_csv(output_path + '/test.csv', index=False)

    logger.info("Data Storing Completed - End of Preprocessing")


if __name__ == "__main__":
    main(opt["--input_path"], opt["--sep"], opt["--test_size"], opt["--random_state"], opt["--output_path"])