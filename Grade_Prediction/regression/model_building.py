# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 12:19:17 2025

@author: meerv
"""

import numpy as np
import pandas as pd

# Machine learning models
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from utils import split_and_scale_data


def machine_learning(models, data, target_variable, test_size=0.2, random_state=42):
    """
   Function to evaluate multiple machine learning models.
   It splits the data, trains the models, and evaluates them using MAE, MSE, and RMSE metrics.
   
   Args:
   models (list): List of machine learning models to be trained.
   data (pd.DataFrame): The dataset containing features and target.
   target_variable (str): The target variable for prediction (default: 'average_grade').
   test_size (float): Proportion of the dataset to include in the test split (default: 0.2).
   random_state (int): Random state for reproducibility (default: 42).
   
   Returns:
   dict: Dictionary containing performance metrics for each model.
   """
    # List of models that require scaling (e.g., Linear Regression, SVM)
    models_needing_scaling = ['LinearRegression', 'SVR', 'Lasso']
    # Store results for each model
    results = {}
    # Loop through the models
    for model in models:
        # Check if the model requires standardization
        model_name = model.__class__.__name__
        
        if model_name in models_needing_scaling:
            X_train, X_test, y_train, y_test,feature_names,scaler = split_and_scale_data(data, target_variable='average_grade', test_size=0.2, random_state=42, scale=True)
    
        else:
            X_train, X_test, y_train, y_test, feature_names,scaler = split_and_scale_data(data, target_variable='average_grade', test_size=0.2, random_state=42, scale=False)
        
        model.fit(X_train, y_train)
        # Predict on the test set
        y_pred = model.predict(X_test)
        # Calculate MAE, MSE, and R2
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        RMSE = np.sqrt(mse)
        
        # Store the results in a dictionary with model name
        results[model_name] = {'MAE': mae, 'MSE': mse, 'RMSE': RMSE}
    
    results = sorted(results.items(), key=lambda x: x[1]['RMSE'])
    return dict(results)





class RandomForestModelTrainer:
    def __init__(self, data, target_variable, test_size=0.2, random_state=42):
        self.data = data
        self.target_variable = target_variable
        self.test_size = test_size
        self.random_state = random_state
        self.rf_models = {}  # Dictionary to store trained models
    
    def train_models(self, number_of_features):
        """Trains Random Forest models for different feature subset sizes."""
        # Train an initial model to get feature importances
        X_train, X_test, y_train, y_test, feature_names, scaler = split_and_scale_data(
            self.data, target_variable=self.target_variable, test_size=self.test_size, random_state=self.random_state, scale=False
        )

        clf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        clf.fit(X_train, y_train)

        # Compute feature importance ranking
        feature_importances = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=True)
        
        for i in number_of_features:
            print(f"\n=== Training Model with Top {i} Features ===")

            # Select the top i features
            top_features = feature_importances.nlargest(i).index.tolist()
            data_selected = self.data[top_features + [self.target_variable]]  # Store selected features in dataset

            # Split and scale data using the existing function
            X_train, X_test, y_train, y_test, feature_names, scaler = split_and_scale_data(
                data_selected, target_variable=self.target_variable, test_size=self.test_size, random_state=self.random_state, scale=False
            )

            # Define hyperparameter space
            param_dist = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }

            # Perform Randomized Search
            random_search = RandomizedSearchCV(
                RandomForestRegressor(random_state=self.random_state),
                param_distributions=param_dist, 
                n_iter=10,
                cv=5, 
                n_jobs=-1, 
                random_state=self.random_state, 
                scoring='neg_root_mean_squared_error'
            )
            random_search.fit(X_train, y_train)

            # Best hyperparameters
            best_params = random_search.best_params_
            print(f"Best hyperparameters for {i} features: {best_params}")

            # Train model with best parameters
            best_clf = RandomForestRegressor(**best_params, random_state=self.random_state)
            best_clf.fit(X_train, y_train)

            # Make predictions and calculate RMSE
            y_pred = best_clf.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            RMSE = np.sqrt(mse)

            # Store results
            self.rf_models[f'Top_{i}_features'] = {
                'num_features': i,
                'RF Model': best_clf,
                'parameters': best_params,
                'RMSE': RMSE,
                'selected_data': data_selected,
                'X_train':X_train,
                "X_test":X_test,
                "y_train":y_train,
                "y_test":y_test,
                "y_pred":y_pred
            }
            
            print(f"Initial RMSE for the top {i} features: {RMSE:.4f}")
            print("=" * 50)
            
    def predict(self, new_data):
        
        """
        Predicts using the best model (lowest RMSE).
        Args:
        - new_data (pd.DataFrame): New data for prediction.     
        Returns:
        - np.array: Predicted values.
        """
        
        if not self.rf_models:
            raise ValueError("No trained models found. Please train models first.")

        # Select the best model based on RMSE
        best_model_data = min(self.rf_models.values(), key=lambda x: x["RMSE"])
        best_model = best_model_data["RF Model"]
        best_features = best_model_data["selected_features"]
        scaler = best_model_data["scaler"]

        # Ensure new_data has the correct features
        new_data = new_data[best_features]

        # Scale the new data
        new_data_scaled = scaler.transform(new_data)

        # Predict using the best model
        return best_model.predict(new_data_scaled)

    def get_results(self):
        """Returns all trained models with their performance metrics and datasets."""
        return self.rf_models



