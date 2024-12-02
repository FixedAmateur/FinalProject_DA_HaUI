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
        try:
            df.to_csv(file)
            print(f"File '{file}' created successfully.")
        except Exception as e:
            print(f"An error occurred while creating the file: {e}")

# 1.2. Reading csv file
def read_csv_file(file_name):
    try:
        return pd.read_csv(file_name +'.csv')
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None
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
def data_clean(df, df_name):

    df = df.dropna()
    df_category = df.select_dtypes(include=['object', 'string'])
    print(f"Categorical variables of {df_name}:", list(df_category.columns), '\n')
    if list(df_category.columns):
        for column in df_category.columns:
            df[column] = LabelEncoder().fit_transform(df[column])
        print(f"\nCategorical variables encoded {df_name}:\n", df, '\n')
    else:
        print(f"\n{df_name} has no categorical variables")
    return df

df = data_clean(df, file_name)

# 3.2. Descriptive table
descriptive_dir = "Descriptive"
os.makedirs(descriptive_dir, exist_ok=True)

df_descriptive = df.describe() # create but not save as file
print(f"Descriptive of {file_name}\n", df_descriptive,'\n')
save_df(df_descriptive, f"{descriptive_dir}/Descriptive_{file_name}", "csv")

# 3.3. Histogram chart
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

os.makedirs("Boxplot",exist_ok=True)
boxplots = boxplot(df, file_name)
for fig in boxplots:
    fig.savefig(f"Boxplot/{fig.axes[0].get_title()}.png")

# 3.4. Save preprocessed file
save_df(df, f"Preprocessed_{file_name}", "csv")


# 4. TRAINING MULTIPLE LINEAR REGRESSION MODEL
# 4.1. Identify features and target variables
target_var = 'Performance Index'
features = [x for x in list(df.columns) if x != target_var]
df_y = df[target_var]
df_x = df[features]

print(f"\n{file_name}'s feature columns:\n", features, '\n')
print(f"\n{file_name}'s target variable:\n", [target_var], '\n')

# 4.2. Split train and test dataframe
evaluation_dir = "Model evaluation"
split_dir = "Split file"
df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42)
os.makedirs(f"{evaluation_dir}/{split_dir}", exist_ok=True)
save_df(df_x_train, f"{evaluation_dir}/{split_dir}/Split_features_of_{file_name}_for_training", "csv")
save_df(df_x_test, f"{evaluation_dir}/{split_dir}/Split_features_of_{file_name}_for_testing","csv")
save_df(df_y_train, f"{evaluation_dir}/{split_dir}/Split_target_of_{file_name}_for_training","csv")
save_df(df_y_test, f"{evaluation_dir}/{split_dir}/Split_target_of_{file_name}_for_testing", "csv")

# 4.3. Create model with splitted dataframes for training
lr_model = LinearRegression().fit(df_x_train, df_y_train)
print("Model created and trained!")

# 4.4. Make prediction on splitted test dataframe
df_y_predicted = pd.DataFrame(lr_model.predict(df_x_test), columns=["Predicted Performance Index"])
save_df(df_y_predicted,f"{evaluation_dir}/{split_dir}/Prediction_of_{file_name}_split_features","csv" )

# 4.4. Evaluate model

os.makedirs(evaluation_dir, exist_ok=True)
def evaluate_lr_model(lr_model, evaluation_dir, x_test, y_test, y_pred):
    evaluation = {
        "RMSE" : np.round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 4),
        "R2" :  np.round(lr_model.score(x_test, y_test), 4)
    }
    df_evaluation = pd.DataFrame(evaluation.items()).transpose()
    save_df(df_evaluation, f"{evaluation_dir}/model_evaluation", "csv")
    return df_evaluation

df_evaluation = evaluate_lr_model(lr_model, "Model evaluation", df_x_test, df_y_test, df_y_predicted)
print("Model evaluation:\n",df_evaluation)

# 4.5. Draw scatterplot
def scatter(x_test, y_test, y_prediction):
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

os.makedirs(f"{evaluation_dir}/Scatter", exist_ok=True)
scatters = scatter(df_x_test, df_y_test, df_y_predicted)
for fig in scatters:
    fig.savefig(f"{evaluation_dir}/Scatter/{fig.axes[0].get_title()}.png")


# 5. MAKING PREDICTION WITH INPUT CSV FILES
print("MAKE PREDICTION WITH INPUT CSV FILES")
# 5.1. Create prediction dataframe from input csv file
def create_prediction_df(test_name, lr_model, features, target_name):
    df_prediction = read_csv_file(test_name)
    df_prediction = data_clean(df_prediction, test_name)
    x = df_prediction[features]
    print(x)
    if x.isnull().sum().sum() > 0:
        print("Warning: The input contains NaN values after cleaning.")
        return None
    y = np.around(lr_model.predict(x), decimals=2)
    df_prediction[target_name] = y
    return df_prediction

# 5.2. Identify tests folder and create result folder
test_dir = 'Tests'
result_dir = 'Results'
os.makedirs(result_dir,exist_ok=True)

# 5.3. Create prediction files
for test in Path(test_dir).iterdir():
    if test.is_file():
        test_name = test.name.split('.')[0]
        df_predicted = create_prediction_df(f'{test_dir}/{test_name}', lr_model, features, 'Predicted Performance Index')
        print(f"{test_name} prediction data:\n",df_predicted)
        save_df(df_predicted, f"{result_dir}/{test_name}_Prediction", "csv")



