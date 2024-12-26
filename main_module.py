import Tuan_module as T
import Nam_module as N
import Hieu_module as H

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

from main_Hieu import evaluation_dir, df_y_predicted

training_file = "Student_Performance"
root_dir = ""
test_dir = "Tests"

# Read training csv
df = T.read_csv_file(training_file)

# Print training csv
T.print_df(df, training_file)

# Preprocess Training data
df = H.preprocess_data(df, training_file)
H.save_preprocessed_data(df, training_file)

# Data Descriptive
T.save_descriptive(df, training_file)

# Draw and save histogram
T.save_histogram(df, training_file)

# Draw and save boxplot
T.save_boxplot(df, training_file)

# Draw and save pie chart
H.piechart(df, training_file)

# Draw and save heatmap
N.heatmap(df, training_file)


# Dependent and independent variables
target = "Performance Index"
df_x, df_y, features = T.identify_variables_df(df, target, training_file)

# split df into train and test file
evaluation_dir = "Model evaluation"
split_dir = "Split file"
df_x_train, df_x_test, df_y_train, df_y_test = T.save_split_train_test(df_x, df_y, training_file, evaluation_dir, split_dir)

# Draw and save scatter
T.save_scatter(df_x_test, df_y_test, evaluation_dir)


# Create and train linear regression model
lr_model = T.train_lr_model(df_x_train, df_y_train)

# Train split file for testing
df_y_predicted = T.predict_split_training_file(df_x_test, lr_model, evaluation_dir, split_dir, training_file)

# Write and save regression equation
T.regression_equation(lr_model, features, target, evaluation_dir)

# Evaluate model
N.evaluate_lr_model(lr_model, evaluation_dir, df_x_test, df_y_test, df_y_predicted)

# Predict data from input csv and save the prediction
T.save_predict_folder(lr_model, features, target, test_dir)
