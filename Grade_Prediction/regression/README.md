# Methodology for Regression 
## 1) Data Preprocessing
In this step, I focus on cleaning, transforming, and preparing the raw data for further analysis and model training. The data preprocessing is carried out using the `data_preprocessing.py` module. Below are the detailed steps followed during preprocessing:

#### **1.1) Load Data**
The first step is to load the raw datasets from a URL or manually from local CSV files. The function `load_data()` handles this process by either downloading the data using the Kaggle API or loading local files if specified. In our case, the two CSV files, `student-mat.csv` and `student-por.csv`, are loaded.

#### **1.2) Merge Datasets**
The datasets from `student-mat.csv` and `student-por.csv` are combined into a single dataframe using the `merge_datasets()` function. This step ensures we have all relevant data combined for further analysis. If the datasets have the same columns, they are concatenated, otherwise, a warning is issued.

#### **1.3) Exploratory Data Analysis (EDA)**
We perform exploratory data analysis (EDA) to understand the structure and patterns in the dataset:

- **Data Overview:** We start by reviewing the basic structure of the data and check the types of variables and missing values.
- **Summary Statistics:** Descriptive statistics such as mean, median, and standard deviation for numerical variables are calculated.
- **Categorical Variables Distribution:** We examine the distribution of categorical variables and their frequencies to understand their behavior.

The `PerformEDA` class is used to perform all these steps.

#### **1.4) Add Average Grade**
To simplify the analysis, we compute an average grade, `average_grade`, by combining the three grades: G1, G2, and G3. These individual grade columns are then dropped to retain only the average grade in the final dataset.

#### **1.5) Visualizations**
We plot histograms and boxplots for numerical columns, such as `age`, `absences`, and `average_grade`, to understand their distribution. This visualization helps to identify potential data issues like skewness or outliers.

#### **1.6) Remove Outliers**
Outliers are removed using the Interquartile Range (IQR) method. For example, we remove extreme values from the `absences` column to ensure our data is more consistent and does not introduce bias.

#### **1.7) Feature Encoding**
Categorical variables are encoded using custom encoding from the `utils.py` module. This ensures that categorical variables are appropriately represented for machine learning models.

#### **1.8) Correlation Analysis**
A correlation heatmap is generated to examine the relationships between numeric variables. This helps identify potential multicollinearity issues and guides feature selection for model training.

The final dataset after preprocessing is returned and ready for use in model training.

## 2) ML Model Selection
In this step, I evaluate and select the best machine learning models for predicting the target variable. 
I use multiple models, train them, and compare their performance based on different metrics like Mean Absolute Error (MAE),
Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). Below are the steps followed for model selection:

#### **2.1) Model Selection**
We select a variety of regression models to assess their performance. The models included in the evaluation are:

- **RandomForestRegressor:** A robust ensemble model that creates multiple decision trees and averages their predictions.
- **XGBRegressor:** A gradient boosting machine that is widely used for its high performance in regression tasks.
- **Lasso (L1 Regularization):** A linear regression model with L1 regularization to enforce sparsity in feature selection.
- **LinearRegression:** A classic linear regression model for simple predictive tasks.
- **SVR (Support Vector Regression):** A non-linear model that uses support vectors for regression tasks.

These models were chosen based on their suitability for both simple and complex regression tasks.

#### **2.2) Model Training and Evaluation**
The `machine_learning()` function is used to train the models and evaluate their performance. This function:

- **Data Splitting:** Splits the dataset into training and testing sets using a test size of 20% (default value). 
- **Scaling:** Some models, such as `LinearRegression`, `SVR`, and `Lasso`, require feature scaling, so the `split_and_scale_data()` function is used for this purpose.
- **Model Training:** Each model is trained on the training data.
- **Model Evaluation:** After training, the models make predictions on the test set, and the following evaluation metrics are calculated:
  - **MAE (Mean Absolute Error)**
  - **MSE (Mean Squared Error)**
  - **RMSE (Root Mean Squared Error)**

The results are then stored in a dictionary, which is sorted by RMSE values to compare the models.
Here is an example of how the results are printed after evaluation:

```python
# Example output format:
for model_name, metrics in results.items():
    print(f"\n{model_name} Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
```
## 3) Feature Selection 
### Overview
Feature selection is an essential step in optimizing machine learning models. It helps in identifying the most important features that contribute to the model’s performance, making the model more efficient and interpretable. In this project, we use **Random Forest Regressor** to perform feature selection by iteratively removing the least important features and evaluating the model's performance after each removal. This process continues until the optimal set of features is determined.

### Feature Selection Process
1. **Data Splitting**: 
   The dataset is split into training and testing sets (80% for training and 20% for testing). Feature scaling is applied based on the model's requirements.
   
2. **Training Random Forest Model**:
   The **RandomForestRegressor** model is initially trained using all available features to compute feature importances.
   
3. **Iterative Feature Removal**:
   In each iteration, the least important feature (determined by the feature importances) is removed. The model is retrained using the remaining features, and its performance is evaluated using **Root Mean Squared Error (RMSE)** on the test set.
   
4. **Stopping Criteria**:
   The process stops when only one feature remains. The number of features and the corresponding RMSE values are tracked throughout the process to identify the optimal set of features.
   
5. **Optimal Features**:
   After the iterative process, the features that provide the best performance (lowest RMSE) are retained for model training.


## 4) Model Training and Evaluation with selected features


## 5) Visualization 
