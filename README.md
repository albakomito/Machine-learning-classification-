# Unit_11_Assignment
Credit Risk
-------------------------------------------------------------------
In this assignment, we evaluate the credit risk of peer-to-peer borrower using Resampling (credit_risk_resampling.ipynb) and  Ensemble Learning (credit_risk_ensemble.ipynb) models. 

## Resampling (credit_risk_resampling.ipynb)
### Initial Imports (Ln 1-2) 
We start by importing the libraries we will use from numpy, pandas, pathlib, and sklearn. 

### Read CSV & Split the Data into Training & Testing (Ln 4-7) 
We read in the CSV file, preview the data, create our features and target, check the balance of our target values, and split the X and y features into X_train, X_test, y_train and y_test. 

### Data Pre-Processing (Ln 8-10)
We create the StandardScaler instance, fit the Standard Scaler with the training data, and Scale the training and testing data. 

### Simple Logistic Regression (Ln 11-14) 
We import LogisticRegression from sklearn and fit the logistic regression model (with X_train_scaled), calculate the balanced accuracy score, display the confusion matrix, and print the imbalanced classification report. 

### Oversampling (Ln 15-24)
We use Naive Random Oversampling and SMOTE to train logistic regression models with our resampled data, and follow the steps from the simple logistic regression to calculate the balanced accuracy score, display the confusion matrix and print the imbalanced classification report. 

### Undersampling (Ln 25-29)
We use ClusterCentroids to undersample our data, fit a logistic regression model (with the undersampled data) and calculate the measures in the Oversampling kernels. 

### Combination Sampling (Ln 30-34) 
We resample the data using SMOTEENN, fit the new logistic regression model and calculate the same measures as those in the Undersampling and Oversampling kernels. 

---------------------------------------------------------------------------------------------------------------------------------------------------------------
## Ensemble Learning (credit_risk_ensemble.ipynb)

### Initial Imports (Ln 1-3)
We import the various libraries that we'll be using. 

### Read CSV file, Basic Data Cleaning, Split the data (Ln 4-8) 
We create our features and target, check the balance of our target values, and split the X and y into X_train, X_test, y_train, y_test. 

### Data Pre-processing (Ln 9-11) 
We create the StandardScaler instance, fit the Standard Scaler with the training data, and scale the training and testing data. 

### Balanced Random Forest Classifier (Ln 12-15)
We resample the training data with the BalancedRandomForestClassifier, calculate the balanced accuracy score, display the confusion matrix,  print the imbalanced classification report and list the features sorted in descending order by feature importance. 

### Easy Ensemble Classifier  (Ln 16-20) 
We train the Classifier using EasyEnsembleClassifier from imblearn, calculate the balanced accuracy score, display the confusion matrix and print the imbalanced classification report. 
