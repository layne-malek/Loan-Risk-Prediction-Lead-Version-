import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE
from sklearn.preprocessing import SplineTransformer
from sklearn.calibration import CalibratedClassifierCV


#Schedule for project:
# 1. Introduction and python/numpy basics
# 2. Data loading and exploration
# 3. Data preprocessing and feature engineering
# 4. Model 1 training
# 5. Model 2 training?
# 6. Model evaluation and visualization
# 7. Hyperparameter tuning and model selection
# 8. Wrap-up and make a website!

#TO DO:
# 1. Add a function to load data from a CSV file
# 2. Implement a function to preprocess the data
# 3. Create a function to train logistic regression model
# 4. Create a function to train a kernel ridge regression or svm model
# 4. Implement a function to evaluate the model
# 5. Add a function to visualize the results




# Function to train logistic regression model
def train_model(X: npt.NDArray, y: npt.NDArray):
    """
    Parameters:
    X (npt.NDArray): Feature matrix.
    y (npt.NDArray): Target vector.
    
    Returns:
    model: Trained log regrression model.
    """
    #May need to use k-stratified cross-validation depending on the dataset
    # Initialize the logistic regression model
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    
    # Could use RFE for feature selection
    #rfe = RFE(estimator=model, n_features_to_select=10)
    #rfe.fit(X, y)
    
    # Fit the model with selected features
    model.fit(X[:, model.support_], y)

    #Can use model.coef_ or model.intercept_ for students to explore/visualize the regression model
    
    return model

#Function to evaluate the trained model
def evaluate_model(model, X: npt.NDArray, y: npt.NDArray):
    """
    Parameters:
    model: Trained logistic regression model.
    X (npt.NDArray): Feature matrix.
    y (npt.NDArray): Target vector.
    
    Returns:
    dict: Evaluation metrics.
    """
    # Predict probabilities
    y_pred = model.predict_proba(X)[:, 1]
    
    # Calculate ROC AUC score
    roc_auc = metrics.roc_auc_score(y, y_pred)
    
    # Calculate accuracy
    accuracy = metrics.accuracy_score(y, model.predict(X))

    # Calculate precision

    # Calculate other metrics
    
    return {'roc_auc': roc_auc, 'accuracy': accuracy}

def visualize_results(y_true: npt.NDArray, y_pred: npt.NDArray):
    """
    Parameters:
    y_true (npt.NDArray): True labels.
    y_pred (npt.NDArray): Predicted labels.
    
    Returns:
    None: Displays the ROC curve and confusion matrix.
    """
    # Plot ROC curve

    # Plot confusion matrix

    #
