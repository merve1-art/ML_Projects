# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 15:12:12 2025

@author: meerv
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def custom_encoding(df):
    """
    Custom encoding function to encode categorical variables in the dataset.
    It applies one-hot encoding for both binary and nominal categorical columns.
    
    Args:
    df (pd.DataFrame): The dataframe to encode.
    
    Returns:
    df (pd.DataFrame): The encoded dataframe with all categorical variables transformed.
    """
    # Iterate through all columns in the dataset
    for col in df.columns:
        if df[col].dtype == 'object':  # For categorical columns (binary or nominal)
            if df[col].nunique() == 2:  # Binary columns (e.g., 'Yes'/'No', 'M'/'F')
                df = pd.get_dummies(df, columns=[col], drop_first=True)  # One-hot encoding for binary columns
            else:  # Nominal columns (more than 2 unique values)
                df = pd.get_dummies(df, columns=[col], drop_first=False)  # One-hot encoding for nominal columns
    # Ensure that all columns are converted to integers after encoding
    df = df.astype(int, errors='ignore')  # Converts all columns to integers (be mindful of non-binary columns)

    return df

def custom_decoding(data):
    data = data.copy()  # Prevent modification of the original DataFrame

    # Find all columns containing "_"
    columns_with_underscore = data.columns[data.columns.str.contains("_")].tolist()

    for col in columns_with_underscore:
        unique_values = sorted(data[col].unique())  # Get unique values, sorted for consistency

        if len(unique_values) == 2 and set(unique_values) == {0, 1}:  # Ensure binary values
            suffix = col.split("_", 1)[1] if "_" in col else col  # Extract suffix
            
            if suffix == "yes":
                # Assign "yes" to 1s and "no" to 0s
                data[col] = data[col].apply(lambda x: "yes" if x == 1 else "no")
            elif suffix == "no":
                # Assign "no" to 1s and "yes" to 0s (opposite of "yes")
                data[col] = data[col].apply(lambda x: "no" if x == 1 else "yes")
            else:
                # Assign the suffix to 1s and "not suffix" to 0s
                data[col] = data[col].apply(lambda x: suffix if x == 1 else f"not {suffix}")

    return data




def split_and_scale_data(data, target_variable, test_size=0.2, random_state=42, scale=True):
    """
    Split data into training and test sets and apply scaling if needed.
    
    Args:
    - data (pd.DataFrame): The dataset.
    - target_variable (str): The target variable for prediction.
    - test_size (float): Proportion of dataset to be used for testing.
    - random_state (int): Random state for reproducibility.
    - scale (bool): Whether to scale the data or not.
    
    Returns:
    - X_train, X_test, y_train, y_test: The split and scaled datasets.
    """
    X = data.drop(columns=[target_variable])
    y = data[target_variable]
    
    feature_names=X.columns
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = None
    
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, feature_names, scaler
