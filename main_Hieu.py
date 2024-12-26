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

# 1. NECESSITIES
# 1.1. Saving dataframe as in formatted file
def save_df(df, file_name, format):
    # -----------------------------------------------------------------------
    # Kiem tra dieu kien xem file do co hop le khong
    if format != 'csv':
        raise ValueError("Only 'csv' format is supported!")
    # -----------------------------------------------------------------------


    # Tạo đường dẫn tệp, filename: tên tệp, format: định dạng tệp
    file = f"{file_name}.{format}"

    # Kiểm tra xem tệp có tồn tại hay không
    if os.path.exists(file):
        # lệnh try chạy khi điều kiện đúng, nếu k có lỗi thì bỏ qua except, có lỗi thì chuyển qua except
        try:
            # Xóa tệp cũ và tạo tệp mới
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

# 1.3. điều chỉnh cách hiển thị số thực trong df của pandas ( cài đặt độ chính xác )
pd.options.display.float_format = '{:.2f}'.format



# 2. READING TRAINING DATA
print("READING TRAINING DATA")
# 2.1. Read student performance training data into pandas dataframe
file_name = 'Student_Performance'
df = read_csv_file(file_name)


# 2.2. Print dataframe
print(f"Columns of {file_name}:", list(df.columns), '\n')
# hàm columns kết hợp với df để đưa ra giá trị các cột dưới dạng Index
# list(df.columns) để chuyển các giá trị về dạng list, giúp hiển thị dễ hiểu hơn
print('\n' + file_name + ":\n", df, '\n')



# 3. DATA PREPROCESSING
#from sklearn.preprocessing import LabelEncoder, StandardScaler



print("PREPROCESS TRAINING DATA")

def preprocess_data(df, df_name, verbose=True):
    # 1. Xóa các giá trị thiếu (Missing Values)
    if df is None or df.empty:
        print(f"Warning: DataFrame {df_name} is empty or None!")
        return None

    if verbose:
        print(f"Initial shape of {df_name}: {df.shape}")
    cleaned_df = df.dropna()
    if verbose:
        print(f"Shape after removing missing values: {cleaned_df.shape}\n")

    # 2. Xử lý dữ liệu phân loại (Categorical Data)
    categorical_columns = cleaned_df.select_dtypes(include=['object', 'string']).columns
    if verbose:
        print(f"Categorical variables in {df_name}: {list(categorical_columns)}\n")

    if len(categorical_columns) > 0:
        label_encoder = LabelEncoder()
        cleaned_df[categorical_columns] = cleaned_df[categorical_columns].apply(
            lambda col: label_encoder.fit_transform(col))
        if verbose:
            print(f"Categorical variables encoded in {df_name}:\n{cleaned_df[categorical_columns]}\n")
    else:
        if verbose:
            print(f"{df_name} has no categorical variables to encode.\n")

    # 3. Loại bỏ các giá trị ngoại lệ (Outliers)
    numerical_columns = cleaned_df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_columns:
        q1 = cleaned_df[col].quantile(0.25)
        q3 = cleaned_df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    if verbose:
        print(f"Shape after removing outliers: {cleaned_df.shape}\n")

    # 4. Chuẩn hóa dữ liệu số (Normalization/Standardization)
    scaler = StandardScaler()
    cleaned_df[numerical_columns] = scaler.fit_transform(cleaned_df[numerical_columns].copy())
    if verbose:
        print(f"Numerical data after standardization:\n{cleaned_df[numerical_columns]}\n")

    # 5. Loại bỏ các cột không cần thiết hoặc dư thừa
    redundant_columns = [col for col in cleaned_df.columns if cleaned_df[col].nunique() == 1]
    if redundant_columns:
        cleaned_df.drop(columns=redundant_columns, inplace=True)
        if verbose:
            print(f"Dropped redundant columns: {redundant_columns}\n")

    if verbose:
        print(f"Final shape of {df_name}: {cleaned_df.shape}\n")

    return cleaned_df

data_cleaned = preprocess_data(df, file_name, verbose=True)


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


# Pie chart
def piechart(df, file_name):
    pies = []
    for column in df:
        fig = plt.figure(figsize=(10, 6))

        # Đếm số lần xuất hiện của các giá trị trong cột để vẽ biểu đồ pie
        value_counts = df[column].value_counts()

        # Vẽ pie chart với các điều chỉnh để nhãn không bị đè lên nhau
        wedges, texts, autotexts = plt.pie(
            value_counts,
            labels=value_counts.index,
            autopct='%1.1f%%',
            startangle=140,
            pctdistance=0.8,  # Khoảng cách phần trăm
            labeldistance=1.17,  # Khoảng cách nhãn ra ngoài
            wedgeprops={'edgecolor': 'black'}  # Thêm đường viền cho các lát cắt
        )

        # Điều chỉnh font size của nhãn và phần trăm
        for text in texts:
            text.set_fontsize(5)
        for autotext in autotexts:
            autotext.set_fontsize(3)

        plt.title(f"Pie chart of {column} in {file_name}")

        # Lưu hình ảnh vào thư mục
        os.makedirs("Pie chart", exist_ok=True)
        plt.savefig(f"Pie chart/{column}_pie_chart.png")

        # Hiển thị biểu đồ
        # plt.show()
        #
        # # Đóng figure để tránh vẽ lại các hình ảnh cũ
        # plt.close(fig)

        pies.append(fig)

    return pies


pie_charts = piechart(df, 'Student_Performance.csv')


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


# 4. TRAINING MULTIPLE LINEAR REGRESSION MODEL
# 4.1. Identify features and target variables
target_var = 'Performance Index'
features = [x for x in list(df.columns) if x != target_var]
df_y = df[target_var]
df_x = df[features]

print(f"\n{file_name}'s feature columns:\n", features, '\n')
print(f"\n{file_name}'s target variable:\n", [target_var], '\n')

# Chỉ giữ các cột số và xử lý giá trị thiếu
df_x = df_x.select_dtypes(include=[np.number]).fillna(0)
df_y = df_y.fillna(0)

# 4.2. Split train and test dataframe
evaluation_dir = "Model evaluation"
split_dir = "Split file"
df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42)
os.makedirs(f"{evaluation_dir}/{split_dir}", exist_ok=True)
save_df(df_x_train, f"{evaluation_dir}/{split_dir}/Split_features_of_{file_name}_for_training", "csv")
save_df(df_x_test, f"{evaluation_dir}/{split_dir}/Split_features_of_{file_name}_for_testing", "csv")
save_df(df_y_train, f"{evaluation_dir}/{split_dir}/Split_target_of_{file_name}_for_training", "csv")
save_df(df_y_test, f"{evaluation_dir}/{split_dir}/Split_target_of_{file_name}_for_testing", "csv")

# 4.3. Create model with splitted dataframes for training
lr_model = LinearRegression().fit(df_x_train, df_y_train)
print("Model created and trained!")

# 4.4. Make prediction on splitted test dataframe
df_y_predicted = pd.DataFrame(np.round(lr_model.predict(df_x_test), 2), columns=["Predicted Performance Index"])
save_df(df_y_predicted, f"{evaluation_dir}/{split_dir}/Prediction_of_{file_name}_split_features", "csv")

# 4.5. Evaluate model
# os.makedirs(evaluation_dir, exist_ok=True)
#
# def evaluate_lr_model(lr_model, evaluation_dir, x_test, y_test, y_pred):
#     y_pred = y_pred.values.flatten()
#     evaluation = {
#         "RMSE": np.round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 4),
#         "R2": np.round(lr_model.score(x_test, y_test), 4)
#     }
#     df_evaluation = pd.DataFrame(evaluation.items()).transpose()
#     save_df(df_evaluation, f"{evaluation_dir}/model_evaluation", "csv")
#     return df_evaluation
#
# df_evaluation = evaluate_lr_model(lr_model, "Model evaluation", df_x_test, df_y_test, df_y_predicted)
# print("Model evaluation:\n", df_evaluation)

def evaluate_lr_model(lr_model, evaluation_dir, x_test, y_test, y_pred, model_name="LinearRegression"):
    # Ensure y_test and y_pred are compatible numpy arrays
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values.flatten()  # Flatten if y_pred is a DataFrame

    # Check for matching lengths
    if len(y_test) != len(y_pred):
        raise ValueError(f"Length mismatch: y_test ({len(y_test)}) and y_pred ({len(y_pred)}) must match.")

    # Calculate evaluation metrics
    rmse = np.round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 4)
    r2 = np.round(lr_model.score(x_test, y_test), 4)

    # Store results in a DataFrame
    evaluation = {
        "Model": model_name,
        "RMSE": rmse,
        "R2": r2
    }
    df_evaluation = pd.DataFrame([evaluation])

    # Ensure the evaluation directory exists
    os.makedirs(evaluation_dir, exist_ok=True)

    # Save evaluation results to a CSV file
    evaluation_path = os.path.join(evaluation_dir, "model_evaluation.csv")
    save_df(df_evaluation, f"{evaluation_dir}/model_evaluation.csv")

    # Print results
    print(f"Model Evaluation ({model_name}):")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")
    print(f"Results saved to {evaluation_path}")

    return df_evaluation

try:
    df_evaluation = evaluate_lr_model(lr_model, "Model evaluation", df_x_test, df_y_test, df_y_predicted)
    print('Model evaluation completed successfully:\n', df_evaluation)
except Exception as e:
    print(f"Error during model evaluation: {e}")


os.makedirs("Heatmap", exist_ok=True)
def heatmap(df, file_name):
    # Check for numeric columns
    numeric_cols = df.select_dtypes(include=['number'])
    if numeric_cols.shape[1] == 0:
        print(f"No numeric columns available in {file_name} for correlation matrix.")
        return

    # Calculate the correlation matrix
    correlation_matrix = numeric_cols.corr()
    print("Correlation matrix:\n", correlation_matrix)

    # Check for NaN values in the correlation matrix
    if correlation_matrix.isnull().values.any():
        print(f"Correlation matrix contains NaN values in {file_name}.")
        return

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=False,  # Turn off default annotations
        cmap='coolwarm',
        linewidths=1,
        cbar_kws={'label': 'Correlation'},
        linecolor='white',
        ax=ax
    )

    # Add bold and large numbers for each cell
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            ax.text(
                j + 0.5, i + 0.5, f"{i},{j}",  # Add index text
                ha='center', va='center',
                fontsize=14, color='black', weight='bold'  # Larger, bold, and blue text
            )

    # Set titles and labels
    plt.title(f"Heatmap for {file_name}", fontsize=16, weight='bold', color='darkblue')
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12, rotation=0)

    # Save and show the heatmap

    plt.savefig(f"Heatmap/{file_name}_heatmap.png", bbox_inches='tight')
    # plt.show()
    # plt.close(fig)

heatmap(df, file_name)


# 4.5. Draw scatterplot
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

os.makedirs(f"{evaluation_dir}/Scatter", exist_ok=True)
scatters = scatter(df_x_test, df_y_test)
for fig in scatters:
    fig.savefig(f"{evaluation_dir}/Scatter/{fig.axes[0].get_title()}.png")


# 5. MAKING PREDICTION WITH INPUT CSV FILES

print("MAKE PREDICTION WITH INPUT CSV FILES")
# def load_csv(file_path):
#     try:
#         # Đọc dữ liệu từ file CSV
#         if not Path(file_path).exists():
#             raise FileNotFoundError(f"File {file_path} does not exist.")
#
#         # Đọc file CSV
#         df = pd.read_csv(file_path)
#
#         # Kiểm tra file có dữ liệu hay không
#         if df.empty:
#             raise ValueError(f"File {file_path} is empty or invalid.")
#
#         return df
#     except Exception as e:
#         print(f"An error occurred while reading the file: {e}")
#         return None

# test_dir = "Tests"
# def create_prediction_df(test_path, lr_model, features, target_name):
#     # Đọc file CSV
#     test_path = Path(test_path)
#     try:
#         df_prediction = pd.read_csv(test_path)
#         if df_prediction.empty:
#             print(f"Error: File {test_path.name} is empty or invalid.")
#             return None
#     except Exception as e:
#         print(f"Error reading file {test_path.name}: {e}")
#         return None
#
#     # Chỉ giữ lại các cột thuộc danh sách features
#     df_prediction = df_prediction[[col for col in df_prediction.columns if col in features]]
#
#     # Kiểm tra nếu còn thiếu cột
#     missing_features = [feature for feature in features if feature not in df_prediction.columns]
#     if missing_features:
#         print(f"Error: Missing features {missing_features} in {test_path.stem}.")
#         return None
#
#     # Mã hóa hoặc loại bỏ các cột không phải số
#     for column in df_prediction.select_dtypes(exclude=[np.number]).columns:
#         if column in features:
#             print(f"Warning: Column {column} is non-numeric. Encoding or dropping it...")
#             # Mã hóa cột (dùng LabelEncoder nếu cần thiết)
#             try:
#                 df_prediction[column] = df_prediction[column].astype(str).factorize()[0]
#             except Exception as e:
#                 print(f"Error encoding column {column}: {e}")
#                 df_prediction = df_prediction.drop(columns=[column])
#
#     # Điền giá trị thiếu
#     df_prediction = df_prediction.fillna(0)
#
#     # Dự đoán
#     try:
#         y = np.around(lr_model.predict(df_prediction), decimals=2)
#         df_prediction[target_name] = y
#     except Exception as e:
#
#         print(f"Error during prediction for {test_path.stem}: {e}")
#         return None
#
#     return df_prediction

def create_prediction_df(test_name, lr_model, features, target_name):
    df_prediction = read_csv_file(test_name)
    try:
        df_prediction = preprocess_data(df_prediction, test_name)
        x = df_prediction[features]
        print(x)
        y = np.around(lr_model.predict(x), decimals=2)
        df_prediction[target_name] = y
    except Exception as e:
        print(f"Error in predicting test file: {test_name}.\nDetailed error: {e}")
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



