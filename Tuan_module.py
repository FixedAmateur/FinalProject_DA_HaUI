import Hieu_module as H
import Nam_module as N

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Saving dataframe as in formatted file
def save_df(df, file_name, format):
    file = f"{file_name}.{format}"
    if os.path.exists(file):
        try:
            os.remove(file)
            df.to_csv(file)
            print(f"File '{file}' has been replaced with the generated file.")
        except Exception as e:
            print(f"An error occurred while replacing the file: {e}")
    else:
        try:
            df.to_csv(file)
            print(f"File '{file}' created successfully.")
        except Exception as e:
            print(f"An error occurred while creating the file: {e}")

# Reading csv file
def read_csv_file(file_name):
    try:
        return pd.read_csv(file_name +'.csv')
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

# Print columns of dataset
def print_df(df, file_name):
    print(f"Columns of {file_name}:", list(df.columns), '\n')
    print('\n' + file_name + ":\n", df, '\n')

# Generate descriptive csv
def save_descriptive(df, file_name):
    descriptive_dir = "Descriptive"
    os.makedirs(descriptive_dir, exist_ok=True)
    df_descriptive = df.describe()  # create but not save as file
    print(f"Descriptive of {file_name}\n", df_descriptive, '\n')
    save_df(df_descriptive, f"{descriptive_dir}/Descriptive_{file_name}",  "csv")
    return df_descriptive


# Split dataframe into dependent and independent variables
def identify_variables_df(df, target_var, file_name):
    df_y = df[target_var]
    features = [x for x in list(df.columns) if x != target_var]
    df_x = df[features]
    print(f"\n{file_name}'s independent variable:\n", features , '\n')
    print(f"\n{file_name}'s target variable:\n", [target_var], '\n')
    return df_x, df_y, features

# Split dataframe into test and train file
def save_split_train_test(df_x, df_y, file_name, evaluation_dir, split_dir):
    #evaluation_dir = "Model evaluation" 
    df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42)

    os.makedirs(f"{evaluation_dir}/{split_dir}", exist_ok=True)

    save_df(df_x_train, f"{evaluation_dir}/{split_dir}/Split_features_of_{file_name}_for_training",  "csv")
    save_df(df_x_test, f"{evaluation_dir}/{split_dir}/Split_features_of_{file_name}_for_testing",  "csv")
    save_df(df_y_train, f"{evaluation_dir}/{split_dir}/Split_target_of_{file_name}_for_training",  "csv")
    save_df(df_y_test,f"{evaluation_dir}/{split_dir}/Split_target_of_{file_name}_for_testing",  "csv")

    return df_x_train, df_x_test, df_y_train, df_y_test

# Train model
def train_lr_model(df_x, df_y) :
    lr_model = LinearRegression().fit(df_x, df_y)
    print("Model created and trained!")
    return lr_model

# Create and save regression_equation
def regression_equation(lr_model, features, target, evaluation_dir):
    coefficients = lr_model.coef_
    intercept = lr_model.intercept_
    equation = f"{target} = {np.round(intercept, 4)}"
    for coef, feature in zip(coefficients, features):
        equation += f" + ({np.round(coef, 4)})*({feature})"

    equa_df = pd.DataFrame({"Regression Equation" : [equation]})

    print(equa_df)
    save_df(equa_df, evaluation_dir + "/Regression equation",  "txt")
    return equation

def predict_split_training_file(x_test, lr_model, evaluation_dir, split_dir, file_name):
    y = pd.DataFrame(np.round(lr_model.predict(x_test), 2), columns=["Performance Index"])
    save_df(y, f"{evaluation_dir}/{split_dir}/Predicted_split_training_file_of_{file_name}", "csv")
    return y

# Prediction with input df
def predict_df(df, df_name , lr_model, features, target):
    try :
        df_prediction = H.preprocess_data(df, df_name)
        x = df_prediction[features]
        print(f"{df_name} prediction data:\n", x)
        y = np.around(lr_model.predict(x), decimals=2)
        df_prediction["Predicted " + target] = y
    except Exception as e:
        print(f"Error in create prediction for {df_name}:\n", e)
    return df_prediction


# Prediction with input test folder with multiple csv files
def save_predict_folder(lr_model, features, target, test_dir):
    result_dir = f'Results'
    os.makedirs(result_dir,exist_ok=True)
    res = []
    for test in Path(test_dir).iterdir():
        if test.is_file():
            test_name = test.name.split('.')[0]
            df_predicted = read_csv_file(f"{test_dir}/{test_name}")
            df_predicted = predict_df(df_predicted, f'{test_dir}/{test_name}', lr_model, features, target)
            print(f"{test_name} prediction data:\n",df_predicted)
            save_df(df_predicted, f"{result_dir}/{test_name}_Prediction","csv")
            res.append(df_predicted)
    return res

# Draw scatterplot
def scatter(x_test, y_test):
    scatters = []
    target = y_test.name
    for column in x_test:
        fig = plt.figure(figsize=(10,8))
        plt.scatter(x_test[column], y_test, color="blue", label="Actual")
        #plt.plot(x_test[column], y_prediction, color="red", label="Predicted")
        plt.title(f"Collaration between {column} and {target}")
        plt.xlabel(column)
        plt.ylabel(target)
        plt.legend()
        scatters.append(fig)

    return scatters

# save scatter
def save_scatter(x_test, y_test, evaluation_dir):
    os.makedirs(f"{evaluation_dir}/Scatter", exist_ok=True)
    scatters = scatter(x_test, y_test)
    for fig in scatters:
        fig.savefig(f"{evaluation_dir}/Scatter/{fig.axes[0].get_title()}.png")

# Histogram chart
def histogram(df, file_name):
    hists = []
    for column in df:
        fig = plt.figure(figsize=(10,6)) #create frame of chart
        sns.histplot(df[column], kde=True) # insert data in frame
        plt.title(f"Distribution of {column} in {file_name}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        hists.append(fig)
    return hists

# Save histogram
def save_histogram(df,file_name):
    histograms = histogram(df, file_name)
    os.makedirs("Histogram",exist_ok=True)
    for fig in histograms:
        fig.savefig(f"Histogram/{fig.axes[0].get_title()}.png")

# 3.4. Boxplot chart
def boxplot(df, file_name):
    boxs = []
    for column in df:
        fig = plt.figure(figsize=(10, 6))
        sns.boxplot(df[column])
        plt.title(f"Boxplot of {column} in {file_name}")
        plt.ylabel(column)
        boxs.append(fig)
    return boxs

# Save boxplot
def save_boxplot(df, file_name):
    os.makedirs("Boxplot",exist_ok=True)
    boxplots = boxplot(df, file_name)
    for fig in boxplots:
        fig.savefig(f"Boxplot/{fig.axes[0].get_title()}.png")