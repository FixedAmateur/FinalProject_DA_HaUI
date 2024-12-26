import Tuan_module as T
import Nam_module as N
import main

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

# Tiền xử lý dữ liệu
def preprocess_data(df, df_name, verbose=True):
    # 1. Xóa các giá trị thiếu (Missing Values)
    if df is None or df.empty:
        print(f"Warning: DataFrame {df_name} is empty or None!")
        return None

    if verbose:
        print(f"Initial shape of {df_name}: {df.shape}")
    preprocessed_df = df.dropna()
    if verbose:
        print(f"Shape after removing missing values: {preprocessed_df.shape}\n")

    # 2. Xử lý dữ liệu phân loại (Categorical Data)
    categorical_columns = preprocessed_df.select_dtypes(include=['object', 'string']).columns
    if verbose:
        print(f"Categorical variables in {df_name}: {list(categorical_columns)}\n")

    if len(categorical_columns) > 0:
        label_encoder = LabelEncoder()
        preprocessed_df[categorical_columns] = preprocessed_df[categorical_columns].apply(
            lambda col: label_encoder.fit_transform(col))
        if verbose:
            print(f"Categorical variables encoded in {df_name}:\n{preprocessed_df[categorical_columns]}\n")
    else:
        if verbose:
            print(f"{df_name} has no categorical variables to encode.\n")


    # # 5. Loại bỏ các cột không cần thiết hoặc dư thừa
    # redundant_columns = [col for col in preprocessed_df.columns if preprocessed_df[col].nunique() == 1]
    # if redundant_columns:
    #     preprocessed_df.drop(columns=redundant_columns, inplace=True)
    #     if verbose:
    #         print(f"Dropped redundant columns: {redundant_columns}\n")
    #
    # if verbose:
    #     print(f"Final shape of {df_name}: {preprocessed_df.shape}\n")



    return preprocessed_df

# Tiền xử lý và lưu dữ liệu
def save_preprocessed_data(df, df_name):
    preprocess_df = preprocess_data(df, df_name)
    T.save_df(preprocess_df, f"Preprocessed_{df_name}",  "csv")
    return preprocess_df


def normalization(df):
    normalized_df = StandardScaler().fit_transform(df.copy())
    print(f"Numerical data after standardization:\n{normalized_df}\n")
    return normalized_df


# Vẽ piechart
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
        os.makedirs(f"Pie chart", exist_ok=True)
        plt.savefig(f"Pie chart/{column}_pie_chart.png")


        pies.append(fig)

    return pies

