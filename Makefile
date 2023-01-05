# Churn Prediction Makefile
# Author: Caesar Wong
# Date: 2022-12-28
# 
# ** please activate the conda environment before using the Make
# This is a makefile for the churn prediction project, there are 3 ways to run the make file:
# 1. `make all`: generate the HTML report and run all the required dependencies files / programs.
# 2. `make <target files>`: only run the specified target files.
# 3. `make clean`: clean all the generated files, images, and report files.


all: results/prediction_output/classification_report.csv

# preprocess data
data/processed/train.csv data/processed/test.csv: data/raw/Churn_Modelling.csv src/preprocessing.py
	python src/preprocessing.py --input_path="data/raw/Churn_Modelling.csv" --sep=',' --test_size=0.3 --random_state=123 --output_path="data/processed"

results/model_output/model_dummy: data/processed/train.csv data/processed/test.csv src/model_training.py
	python src/model_training.py --train="data/processed/train.csv" --target_name="Exited" --out_dir="results/"

results/prediction_output/classification_report.csv: results/model_output/model_lgbm
	python src/model_prediction_interpretation.py --train="data/processed/train.csv" --test="data/processed/test.csv" --target_name="Exited" --target_names="Stayed,Exited" --model_path="results/model_output/" --out_dir="results/prediction_output/"

# # generate html report
# doc/The_Report_of_churn_Prediction.html : doc/The_Report_of_churn_Prediction.Rmd doc/churn_prediction_references.bib results/Confusion_Matrix_logisticRegression.png results/Confusion_Matrix_NaiveBayes.png results/Confusion_Matrix_RandomForestClassifier.png results/PR_curve.png results/ROC_curve.png results/score_on_test.csv results/target_count_bar_plot.png results/correlation_heatmap_plot.pn results/correlation_with_target_plot.png results/gender_density_plot.png
# 	Rscript -e 'rmarkdown::render("doc/The_Report_of_churn_Prediction.Rmd")'

clean:
	rm -f data/processed/*.csv
	rm -f results/model_output/*
	rm -f results/prediction_output/*