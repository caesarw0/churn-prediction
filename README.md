# churn-prediction

![bank](img/bank_graphic.jpg)
Figure 1: Bank Customer Illustration  
Source: Adapted from [1]

## Introduction

Dataset from Kaggle - [bank-customer-churn-prediction](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)

Problem definition
- classification problem with binary target (0: Stayed, 1: Exited)

Best performance
- model: lgbm (tuned)
- macro avg f1 score: 0.77

Implemented using software engineering practice
- logging
- OOP, single class
- modular code
- unit test (future direction)

## Dependencies

Please find more details in the [env/environment.yml](env/environment.yml) for the Python-related dependencies.

## Usage 1

Directly run the [Jupyter Notebook](churn_prediction.ipynb)

## Usage 2

Run Python script using [Makefile](Makefile)

### Run All

To run the whole analysis, run the following command in the root directory:

```
make all
```

It will check whether the [classification report](results/prediction_output/classification_report.csv) exists or not. If the classification report does not exist, the Makefile will run all the dependencies required to generate the report.

### Clean Files

To clean the intermediate and final results including images, CSV files and report, run the following command in the root directory:

```
make clean
```

It will clean all the files under `results/model_output/`, `results/prediction_output/`, and all the CSV files under `data/processed/`.

## Reference

[1] A. Frazier. "Improving Customer Communication at Your Bank." ProcessMaker. Accessed: Dec. 28, 2022. [Online]. Available: https://www.processmaker.com/blog/improve-bank-customer-communication/