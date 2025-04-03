# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 13:36:27 2025

@author: meerv
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from utils import custom_decoding
import matplotlib.colors as mcolors

def plot_number_of_feature_vs_RMSE(feature_counts, rmse_score):
    """
    Plot number of features used in the model and corresponding RMSE.
    
    Args:
   - feature_counts (list): Number of features used at each iteration.
   - rmse_score (list): Corresponding RMSE values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(feature_counts, rmse_score, marker='o', linestyle='-', color='b')
    plt.xlabel("Number of Features")
    plt.ylabel("RMSE")
    plt.title("Feature Selection: Number of Features vs. RMSE")
    plt.gca().invert_xaxis()  # Makes sure decreasing features is left-to-right
    plt.grid(True)
    plt.show()


def plot_feature_importance(model, top_features):
    """
    Plot feature importance for a trained model.
    
    Args:
    - model (model): The trained model.
    - top_features (list): List of features for plotting.
    """
    feature_importances = pd.Series(model.feature_importances_, index=top_features).sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_importances, y=feature_importances.index, palette="viridis")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance in Random Forest Model")
    plt.show()

    



def plot_residuals_vs_predicted(y_test,y_pred):
    residuals = y_test - y_pred  
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7, color="purple")
    plt.axhline(y=0, color="red", linestyle="--")  # Reference line at 0
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Error)")
    plt.title("Residual Plot")
    plt.show()




def plot_actual_vs_predicted(data, X_test, y_test, y_pred, colour_variable=None):
    """
    Plot actual vs predicted values for regression with optional coloring.

    Args:
    - data (pd.DataFrame): Original dataset.
    - X_test (pd.DataFrame): Test set features.
    - y_test (pd.Series): True target values.
    - y_pred (np.ndarray): Predicted target values.
    - colour_variable (list, optional): List of feature names used for color coding.
    """
    X_test_df = custom_decoding(pd.DataFrame(X_test, columns=data.drop(columns=["average_grade"]).columns))

    # Handle the case where no colour_variable is given
    if colour_variable is None:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color="blue")
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs. Predicted Values")
        plt.show()
        return  

    for i in colour_variable: 
        # Check if feature exists in the dataframe
        if i not in data.columns:
            print(f"Warning: {i} is not a column in the dataset!")
            continue

        # Check if the column is categorical or numerical
        is_categorical = data[i].dtype == "object" or data[i].nunique() < 10  # ✅ Handle categorical with few unique values
        is_numerical = np.issubdtype(data[i].dtype, np.number)  

        plt.figure(figsize=(10, 6))

        if is_categorical:
            # Seaborn works well for categorical coloring
            sns.scatterplot(x=y_test, y=y_pred, hue=X_test_df[i].astype(str), palette="viridis", alpha=0.7)  # ✅ Convert to string
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(f"Actual vs. Predicted Values Colored by {i}")
            plt.legend(title=i)
            plt.show()

        elif is_numerical:
            # Use matplotlib for numerical coloring
            norm = mcolors.Normalize(vmin=X_test_df[i].min(), vmax=X_test_df[i].max())  
            cmap = plt.cm.viridis

            scatter = plt.scatter(x=y_test, y=y_pred, c=X_test_df[i], cmap=cmap, norm=norm, alpha=0.7)
            cbar = plt.colorbar(scatter)
            cbar.set_label(f"{i} Feature Value")  # Correctly label the colorbar

            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(f"Actual vs. Predicted Values Colored by {i}")
            plt.show()

        else:
            print(f"Warning: {i} is not a recognized categorical or numerical variable!")


def plot_shap_explainer(model,X_train,X_test):
    # Create SHAP explainer
    explainer = shap.Explainer(model, X_train)
    # Compute SHAP values
    shap_values = explainer(X_test)
    # Summary plot (shows impact on predictions)
    shap.summary_plot(shap_values, X_test)

