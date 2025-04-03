# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 16:15:53 2025

@author: meerv
"""

import pandas as pd
import opendatasets as od
import os
import seaborn as sns
import matplotlib.pyplot as plt
from utils import custom_encoding

def load_data(dataset_url=None, manual=False, base_path=None):
    """
    Loads the data from a URL (using open datasets) or from manual CSV files.
    If `manual` is True, `base_path` must be provided.
    """
    if not manual:
        if dataset_url:
            od.download(dataset_url)
            dataset_name = dataset_url.split("/")[-1]  # Extract dataset folder name
            base_path = os.path.join(os.getcwd(), dataset_name)  # Get full dataset path
        else:
            raise ValueError("dataset_url must be provided when manual is False.")

        df_mat = pd.read_csv(os.path.join(base_path, "student-mat.csv"))
        df_por = pd.read_csv(os.path.join(base_path, "student-por.csv"))
    else:
        if base_path is None:
            raise ValueError("base_path must be provided when manual is True.")
        
        try:
            df_mat = pd.read_csv(os.path.join(base_path, "student-mat.csv"))
            df_por = pd.read_csv(os.path.join(base_path, "student-por.csv"))
        except FileNotFoundError:
            print("CSV files not found in the specified directory.")
            return None
    
    return df_mat, df_por
def merge_datasets(df_mat, df_por):
    """
    Merges the two dataframes and returns the combined dataframe.
    """
    if list(df_mat.columns) == list(df_por.columns):
        df_combined = pd.concat([df_mat, df_por], axis=0, ignore_index=True)
        return df_combined
    else:
        print("Warning: Datasets have different columns. Merging skipped.")
        return None

class PerformEDA:
    def __init__(self, original_data):
        """Initialize with the dataset."""
        self.original_data = original_data.copy()  # Store a copy to prevent modifying the original
    
    def general_eda(self):
        """Perform general exploratory data analysis (EDA)."""
        print("\n--- Data Overview ---")
        self.original_data.info()

        print("\n--- Summary Statistics ---")
        print(self.original_data.describe())

        print("\n--- Categorical Variables Distribution ---")
        categorical_columns = self.original_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            print(f"\n{col} distribution:\n{self.original_data[col].value_counts()}")

    def add_average_grade(self):
        """Compute the average of G1, G2, and G3 as 'average_grade'."""
        self.original_data['average_grade'] = self.original_data[['G1', 'G2', 'G3']].mean(axis=1)
        self.original_data.drop(columns=['G1', 'G2', 'G3'], inplace=True)
    
    def plot_visualizations(self, numeric_columns):
        """Plot histograms and boxplots for numeric variables."""
        self.original_data[numeric_columns].hist(bins=20, edgecolor='black', figsize=(12, 6))
        plt.suptitle('Distribution of Numeric Variables', fontsize=14)
        plt.show()

        plt.figure(figsize=(12, 6))
        for i, col in enumerate(numeric_columns, 1):
            plt.subplot(1, len(numeric_columns), i)
            sns.boxplot(x=self.original_data[col], color='skyblue')
            plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.show()

    def remove_outliers(self, column):
        """Remove outliers from a specified column using the IQR method."""
        Q1 = self.original_data[column].quantile(0.25)
        Q3 = self.original_data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        self.original_data = self.original_data[(self.original_data[column] >= lower_bound) & 
                                                (self.original_data[column] <= upper_bound)]

    def plot_correlation_heatmap(self, numeric_columns):
        """Plot a heatmap showing correlations between numeric variables."""
        corr_matrix = self.original_data[numeric_columns].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title("Correlation Matrix of Numeric Variables", fontsize=14)
        plt.show()

    def perform_eda(self):
        """Performs EDA and preprocessing in sequence."""
        self.general_eda()
        self.add_average_grade()
        
        numeric_columns = ['age', 'absences', 'average_grade']
        self.plot_visualizations(numeric_columns)
        
        self.remove_outliers('absences')
        # Apply custom encoding from utils
        self.original_data = custom_encoding(self.original_data)
        self.plot_correlation_heatmap(numeric_columns)

        return self.original_data  # Return cleaned and processed data





