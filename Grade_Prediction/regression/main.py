# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 15:38:31 2025

@author: meerv
"""
#Import requirements
pip install -r requirements.txt

#ML models
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


#functions from modules 
from data_preprocessing import load_data,merge_datasets,PerformEDA
from model_building import machine_learning,RandomForestModelTrainer
from feature_selection import feature_selection_with_random_forest
from plotting import plot_number_of_feature_vs_RMSE,plot_feature_importance,plot_residuals_vs_predicted,plot_actual_vs_predicted,plot_shap_explainer

#import data 
dataset_url = "https://www.kaggle.com/datasets/uciml/student-alcohol-consumption"
df_mat,df_por=load_data(dataset_url, manual=False)

#merge data
df_combined=merge_datasets(df_mat, df_por)

# Initialize EDA class
eda = PerformEDA(df_combined)

# Run the full EDA pipeline
df_processed = eda.perform_eda()



target_variable='average_grade'


#ML models
models = [RandomForestRegressor(), XGBRegressor(), Lasso(alpha=0.1), LinearRegression(), SVR()]

results=machine_learning(models,df_processed, target_variable, test_size=0.2, random_state=42)

# Print results of all models in a more structured format
for model_name, metrics in results.items():
    print(f"\n{model_name} Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        
        
#feature selection
best_features, feature_counts, rmse_score=feature_selection_with_random_forest(df_processed, 
                                                                target_variable, test_size=0.2,
                                                                random_state=42, scale=False)

#visualization of feature selection
plot_number_of_feature_vs_RMSE(feature_counts, rmse_score)



#Train the model with the selected # of features
trainer = RandomForestModelTrainer(df_processed, target_variable)
trainer.train_models([21,18])
results = trainer.get_results()
print(results.keys())

# Example: Access the dataset used for the model trained with the top 18 features
top_18_data = results['Top_18_features']['selected_data']
model=results['Top_18_features']['RF Model']
X_train=results['Top_18_features']['X_train']
X_test=results['Top_18_features']['X_test']
y_train=results['Top_18_features']['y_train']
y_test=results['Top_18_features']['y_test']
y_pred=results['Top_18_features']['y_pred']

# Display the first few rows of the selected dataset
print(top_18_data.head())


#visualization of model 

#1) featue importance plot 
top_features=top_18_data.drop(columns=[target_variable]).columns
plot_feature_importance(model, top_features)


#2) residuals vs predicted plot
plot_residuals_vs_predicted(y_test,y_pred)


#3) plotting actual vs predicted
plot_actual_vs_predicted(top_18_data,X_test, y_test, y_pred,colour_variable=top_18_data.columns.drop([target_variable]))

#4) shap 
plot_shap_explainer(model,X_train,X_test)


