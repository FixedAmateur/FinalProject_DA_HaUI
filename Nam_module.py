import Tuan_module as T
import Hieu_module as N
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

def evaluate_lr_model(lr_model, evaluation_dir, x_test, y_test, y_pred):
    rmse = np.round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 4)
    r2 = np.round(lr_model.score(x_test, y_test), 4)

    evaluation = {
        "Model" : "Linear Regression",
        "RMSE": rmse,
        "R2": r2
    }

    df_evaluation = pd.DataFrame(evaluation.items(), columns=["Metric", "Value"]).transpose()

    df_evaluation.columns = df_evaluation.iloc[0]
    df_evaluation = df_evaluation.drop(df_evaluation.index[0])

    T.save_df(df_evaluation, f"{evaluation_dir}/model_evaluation", "csv")
    return df_evaluation


def heatmap(df, file_name):
    os.makedirs("Heatmap", exist_ok=True)

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
    return plt




