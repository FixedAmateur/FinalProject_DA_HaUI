import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression


# 1. NECESSITIES
# 1.1. Saving dataframe as in formatted file
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
        df.to_csv(file)
        print(f"File '{file}' created successfully.")

# 1.2. Reading csv file
def read_csv_file(file_name):
    try:
        return pd.read_csv(file_name +'.csv')
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
# 1.3. Set display precision for dataframe
pd.options.display.float_format = '{:.2f}'.format


# 2. READING TRAINING DATA
print("READING TRAINING DATA")
# 2.1. Read student performance training data into pandas dataframe
file_name = 'Student_Performance'
df = read_csv_file(file_name)

# 2.3. Print dataframe
print(f"Columns of {file_name}:", list(df.columns), '\n')
print('\n' + file_name + ":\n", df, '\n')

# 3. DATA PREPROCESSING
print("PREPROCESS TRAINING DATA")
# 3.1. Remove missing values and encode categorical variables
def data_clean(df, file_name):
    df = df.dropna()

    df_category = df.select_dtypes(include=['object', 'string'])
    print(f"Categorical variables of {file_name}:", list(df_category.columns), '\n')
    if list(df_category.columns):
        for column in df_category.columns:
            df[column] = LabelEncoder().fit_transform(df[column])

        print("\nCategorical variables encoded " + file_name + ":\n", df, '\n')

    return df

df = data_clean(df, file_name)

# 3.2. Descriptive table
descriptive_dir = "Descriptive"
os.makedirs(descriptive_dir, exist_ok=True)

df_descriptive = df.describe()
print(f"Descriptive of {file_name}\n", df_descriptive,'\n')
save_df(df_descriptive, f"{descriptive_dir}/Descriptive_{file_name}", "csv")

# 3.3. Histogram chart
def histogram(df, file_name):
    hists = {}
    for column in df:
        fig = plt.figure(figsize=(10,6))
        sns.histplot(df[column], kde=True)
        plt.title(f"Distribution of {column} in {file_name}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        hists[column] = fig
    return hists

histograms = histogram(df, file_name)
os.makedirs("Histogram",exist_ok=True)
for column, fig in histograms.items():
    fig.savefig(f"Histogram/{column}.png")

# 3.4. Boxplot chart
def boxplot(df, file_name):
    boxs = {}
    for column in df:
        fig = plt.figure(figsize=(10, 6))
        sns.boxplot(df[column])
        plt.title(f"Boxplot of {column} in {file_name}")
        plt.ylabel(column)
        boxs[column] = fig
    return boxs

os.makedirs("Boxplot",exist_ok=True)
boxplots = boxplot(df, file_name)
for column, fig in boxplots.items():
    fig.savefig(f"Boxplot/{column}.png")

# 3.4. Save preprocessed file
save_df(df, f"Preprocessed_{file_name}", "csv")

# 3.5. Identify features and target variables
target_var = 'Performance Index'
features = [x for x in list(df.columns) if x != target_var]
df_y = df[target_var]
df_x = df[features]

print(f"\n{file_name}'s feature columns:\n", list(df_x.columns), '\n')
print(f"\n{file_name}'s target variable:\n", [target_var], '\n')


# 4. TRAINING MULTIPLE LINEAR REGRESSION MODEL
lr_model = LinearRegression().fit(df_x, df_y)
print("Model created and trained!")

# 5. MAKING PREDICTION WITH INPUT CSV FILES
print("MAKE PREDICTION WITH INPUT CSV FILES")
# 5.1. Create prediction dataframe from input csv file
def create_prediction_df(test_name, lr_model):
    df_prediction = read_csv_file(test_name)
    data_clean(df_prediction, test_name)
    x = df_prediction.to_numpy()
    y = np.around(lr_model.predict(x), decimals=2)
    df_prediction['Predicted Performance Index'] = y
    return df_prediction

# 5.2. Identify tests folder and create result folder
test_dir = 'Tests'
result_dir = 'Results'
os.makedirs(result_dir,exist_ok=True)

# 5.3. Create prediction files
for test in Path(test_dir).iterdir():
    if test.is_file():
        test_name = test.name.split('.')[0]
        df_predicted = create_prediction_df(f'{test_dir}/{test_name}', lr_model)
        print(f"{test_name} prediction data:\n",df_predicted)
        save_df(df_predicted, f"{result_dir}/{test_name}_Prediction", "csv")



