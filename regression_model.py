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
# 1. Introduction and setup and numpy/python basics
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
def evaluate_model(model, X: npt.NDArray, y_true: npt.NDArray):
    """
    Parameters:
    model: Trained logistic regression model.
    X (npt.NDArray): Feature matrix.
    y (npt.NDArray): Target vector.
    
    Returns:
    dict: Evaluation metrics.
    """
    #Predict probabilities
    y_pred = model.predict_proba(X)[:, 1]
    
    #Calculate AUROC score
    auroc = metrics.roc_auc_score(y_true, y_pred, labels = [-1,1])
    
    #Calculate accuracy
    accuracy = metrics.accuracy_score(y_true, y_pred)

    #Calculate precision
    precision = metrics.precision_score(y_true, y_pred, zero_division = 0.0)
    
    #Calculate recall
    recall = metrics.recall_score(y_true,y_pred)

    #Calculate F1 score
    f1_score = metrics.f1_score(y_true, y_pred)

    #Calculate specificity
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    denom = conf_matrix[0][0]+conf_matrix[0][1]
    if denom == 0:
        denom = 1
    specificity = conf_matrix[0][0]/(denom)
    
    #Calculate average precision
    average_precision = metrics.average_precision_score(y_true, y_pred)
   
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score, 'specificity': specificity, 
            'average_precision': average_precision, 'auroc': auroc}

def visualize_results(y_true: npt.NDArray, y_pred: npt.NDArray):
    """
    Parameters:
    y_true (npt.NDArray): True labels.
    y_pred (npt.NDArray): Predicted labels.
    
    Returns:
    None: Displays the ROC curve and confusion matrix.
    """
    
    # Plot ROC curve

    plt.figure(figsize=(7, 5))

    
    #fpr, tpr, _ = roc_curve(test_df['True'], test_df[model])
    #roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    # Plot confusion matrix

    #
