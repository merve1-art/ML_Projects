# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 17:35:12 2025

@author: meerv
"""
import numpy as np 
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from utils import split_and_scale_data


def feature_selection_with_random_forest(df, target_variable, test_size=0.2, random_state=42, scale=False):
    """
    Perform feature selection using Random Forest by iteratively removing the least important feature.

    Args:
    - df (pd.DataFrame): The dataset including features and the target variable.
    - target_column (str): The name of the target column.
    - test_size (float): The proportion of data to be used for testing.
    - random_state (int): Random state for reproducibility.
    - scale (bool): Whether to scale the features before model training.

    Returns:
    - best_features (list): List of selected features.
    - feature_counts (list): List tracking the number of features left at each step.
    - RMSE_score (list): List tracking RMSE values at each step.
    """
    # Ensure the dataset has enough features
    if df.shape[1] < 2:
       raise ValueError("The dataset must have at least two columns: one feature and the target variable.")


    # Split and scale data
    X_train, X_test, y_train, y_test, feature_names,scaler = split_and_scale_data(df, target_variable, test_size, random_state, scale)

    # Initialize model
    clf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)

    # Store results
    feature_counts = []
    rmse_score = []
    best_features = list(feature_names) # Track current feature set

    # Feature selection loop
    while len(best_features) > 1:
        # Compute feature importances
        feature_importances = pd.Series(clf.feature_importances_, index=best_features).sort_values(ascending=False)

        # Remove the least important feature
        least_important = feature_importances.idxmin()
        best_features.remove(least_important)  

        # Subset dataset with selected features
        X_train_selected =X_train[best_features]
        X_test_selected = X_test[best_features]

        #Retrain model with reduced features
        clf.fit(X_train_selected, y_train)

        #Evaluate model performance
        y_pred = clf.predict(X_test_selected)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Store feature count and RMSE
        feature_counts.append(len(best_features))
        rmse_score.append(rmse)

        print(f"Features left: {len(best_features)} | rmse: {rmse:.4f}")

    return best_features, feature_counts, rmse_score













