# Predicting Student Academic Performance Using Regression and Classification 

**Description:**

This project investigates the factors that influence student academic performance using the "Student Alcohol Consumption"
dataset from Kaggle. It applies both regression and classification techniques to predict student grade categories and 
identifies the key features that contribute to academic success. The regression analysis is detailed in the regression file,
while the classification process is outlined in the classification file. Both approaches are applied to the same dataset,
with various methods explored to optimize the fit for each regression and classification model.
**Regression:**
This analysis covers data preprocessing, including missing value imputation and outlier detection, followed by data encoding.
It involves selecting the appropriate regression technique for modeling, performing feature selection using Random Forest
Regression, and fine-tuning the model's hyperparameters based on the selected features. Additionally, it includes an
interpretation of feature importance to assess their impact on the model.

**Classification:**
The classification analysis involves data preprocessing, model training, and hyperparameter optimization. It also includes 
feature selection to enhance model performance, along with SHAP (SHapley Additive exPlanations) analysis to interpret the
contribution of individual features to the model's predictions.

**Dataset:**

* **Source:** [Student Alcohol Consumption Dataset](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption) from Kaggle.
* **Content:** The dataset contains information about student demographics, social habits, and academic performance, including math and Portuguese language grades.
* **Preprocessing:** The dataset was cleaned, merged, and preprocessed, including outlier removal, feature engineering, and one-hot encoding for categorical variables.

# To import the dataset:
This project utilizes the "Student Alcohol Consumption" dataset from Kaggle. 
You can load the data into the project using one of the following methods: 

### Option 1: Download from the Repository / Manual Download
#### 1) Raw Data: 
You can find the raw datasets in the data/raw/student-alcohol-consumption folder in this repository. Download these files and import the data using the data_preprocessing.py module.

#### 2) Preprocessed Data:
If you want to skip the preprocessing step, you can use the preprocessed and encoded data located in the data/processed folder. In this case, you can start directly with the model_building.py module.

### Option 2: Downloading the Dataset Using the Kaggle API

To download the dataset using the Kaggle API, you need a Kaggle account and an API token. Follow the steps below to set it up:

#### 1. Create a Kaggle Account
If you don't already have one, create an account at [Kaggle](https://www.kaggle.com).

#### 2. Generate a Kaggle API Token
- Log in to your Kaggle account.
- Navigate to your account settings by clicking on your profile picture.
- In the "API" section, click **Create New API Token** to download the `kaggle.json` file containing your API credentials. Make sure to keep this file secure.

### 3. Set Up the Kaggle API
To download the dataset, you'll need to configure the Kaggle API on your machine:
- Install the Kaggle Python library by running the following command in your terminal:
  ```bash
  pip install kaggle
This project utilizes the "Student Alcohol Consumption" dataset from Kaggle. To download this dataset, you'll need a Kaggle account and an API token. 
Follow these steps:
1) Create a Kaggle Account (if you don't have one):
2) Generate a Kaggle API Token:
* Log in to your Kaggle account.
* Go to your account settings (usually by clicking your profile picture).
* In the "API" section, click "Create New API Token".
This downloads a kaggle.json file containing your API credentials. Keep this file safe!
3)Setting Up the Kaggle API:
To download the dataset, you'll need to configure the Kaggle API:
*Install the Kaggle Library: If you don't have the Kaggle library installed, open your terminal or command prompt and run:
pip install kaggle
* Place Your API Token:
Locate the kaggle.json file you downloaded from your Kaggle account settings.
* Move this file to the appropriate directory on your system:
Windows: C:\Users\<Windows-Username>\.kaggle\
macOS/Linux: ~/.kaggle/
If the .kaggle directory doesn't exist, create it.
Then go to the import_dataset file and read the code in option 1 in Python. If the system asks username during data installation from Kaggle,
write your username in Kaggle.




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

1.  **Data Acquisition and Merging:**
    * Downloaded and combined math and Portuguese language student datasets.
2.  **Exploratory Data Analysis (EDA):**
    * Checked data types, missing values, and summary statistics.
    * Created a new "average_grade" feature and calculated the average grade of each student in one semester.
    * Visualized data distributions and identified outliers.
3.  **Data Preprocessing:**
    * Removed outliers using the IQR method.
    * Removed grades for each exam and stuck with the average grade only.
    * One-hot encoded categorical features.
    * Converted grade categories into numerical labels.
    * Handled class imbalance using SMOTE.
4.  **Model Training and Evaluation:**
    * Trained and evaluated multiple classification models (Logistic Regression, Random Forest, SVC, KNN, Decision Tree).
    * Observed low initial accuracy due to grade category distribution.
5.  **K-Means Clustering:**
    * Used K-Means clustering to identify optimal grade categories (K=3).
    * Mapped clusters to "Low," "Medium," and "High" grade categories.
    * Improved model accuracy with the new grade categories.
6.  **Feature Selection:**
    * Performed recursive feature elimination using Random Forest to identify the most important features.
    * Selected top 16 and 36 features for further analysis.
7.  **Hyperparameter Tuning:**
    * Used RandomizedSearchCV to tune hyperparameters for the Random Forest model.
    * Evaluated model performance with the selected features.
8.  **Feature Impact Analysis:**
    * Visualized feature importances using bar plots.
    * Used SHAP values to explain how individual features impact grade predictions.

**Results:**

* The Random Forest Classifier achieved an accuracy of approximately 72% with both 16 and 36 features.
* Feature selection did not significantly impact accuracy, suggesting that the reduced feature set is sufficient.
* Features like "absences," "failures," and "health" were identified as the most influential factors.
* SHAP analysis provided insights into how individual features affect the probability of students falling into different grade categories.

**How to Run the Code:**

1.  Clone the repository: `git clone [repository URL]`
2.  Install required dependencies: `pip install pandas numpy scikit-learn imblearn shap matplotlib seaborn kaggle`
3.  Download the dataset from Kaggle using the Kaggle API and place it in the project directory.
4.  Run the Python script: `python your_script_name.py`

**Files Included:**

* `students_grade_classification.py`: Python script containing the project code.
* `student-mat.csv`: Math student dataset.
* `student-por.csv`: Portuguese student dataset.
* `README.md`: Project documentation.

**Dependencies:**

* pandas
* numpy
* scikit-learn
* imblearn
* shap
* matplotlib
* seaborn
* kaggle

**Author:**

* Merve Tuncer Ozer
* [My GitHub Profile URL] (https://github.com/merve1-art)

**License:**

* [MIT License]

**Future Work:**

* Further investigate and predict the grades with regression models instead of classification.
* Investigate the impact of other features not included in the dataset.
* Develop a web application to visualize the results and provide interactive insights.
