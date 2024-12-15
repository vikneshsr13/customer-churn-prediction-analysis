# Customer Churn Prediction Analysis

This project aims to predict customer churn for a subscription-based business using machine learning techniques. The dataset used is the **Telco Customer Churn dataset**, which contains customer demographics, account information, and service usage data.

## Features
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and feature scaling.
- **Exploratory Data Analysis (EDA)**: Visualizations to understand key factors driving customer churn.
- **Machine Learning**: Random Forest Classifier for predicting churn.
- **Model Evaluation**: Accuracy, confusion matrix, and classification report for performance metrics.
- **Feature Importances**: Insights into the most influential factors affecting churn.

## Project Workflow
1. **Load the Dataset**: Import the Telco Customer Churn dataset into a Pandas DataFrame.
2. **Data Preprocessing**: 
   - Drop irrelevant features (e.g., customer ID).
   - Handle missing values in `TotalCharges`.
   - Encode categorical variables using `LabelEncoder`.
3. **Feature Scaling**: Normalize features using `StandardScaler`.
4. **Train-Test Split**: Split the data into training and testing sets.
5. **Model Building**: Train a `RandomForestClassifier` to predict churn.
6. **Evaluation**: Use metrics like accuracy, confusion matrix, and classification report to assess performance.
7. **Feature Importance**: Visualize which features contribute the most to the model.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/vikneshsr13/Customer-Churn-Prediction-analysis.git


The dataset used in this project is Telco Customer Churn. Download it and save it as telco-customer-churn.csv in the root directory.

Libraries Used

Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Joblib

Results

Accuracy: Achieved an accuracy score of approximately XX% (replace with your score).
Feature Importances: The top 3 features influencing churn were:
Feature A
Feature B
Feature C

