import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import warnings
warnings.filterwarnings("ignore")

def train_model():
    
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')
    
     
    # Logistic Regression model
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)

    # Decision Tree model
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)

    # Predictions
    logistic_predictions = logistic_model.predict(X_test)
    tree_predictions = tree_model.predict(X_test)

    # Evaluate models
    logistic_accuracy = accuracy_score(y_test, logistic_predictions)
    tree_accuracy = accuracy_score(y_test, tree_predictions)

    print("Logistic Regression Accuracy:", logistic_accuracy)
    print("Decision Tree Accuracy:", tree_accuracy)

    # Build K-Nearest Neighbors (K-NN), Support Vector Machine (SVM),
    # Naive Bayesian, Random Forest, and Adaboost


    # Initialize classifiers
    knn_model = KNeighborsClassifier()
    svm_model = SVC()
    naive_bayes_model = GaussianNB()
    random_forest_model  = RandomForestClassifier()
    adaboost_model  = AdaBoostClassifier()

    # Train classifiers
    knn_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    naive_bayes_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)
    adaboost_model.fit(X_train, y_train)

    # Predictions
    knn_pred = knn_model.predict(X_test)
    svm_pred = svm_model.predict(X_test)
    nb_pred = naive_bayes_model.predict(X_test)
    rf_pred = random_forest_model.predict(X_test)
    adaboost_pred = adaboost_model.predict(X_test)

    # Calculate accuracies
    knn_accuracy = accuracy_score(y_test, knn_pred)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    adaboost_accuracy = accuracy_score(y_test, adaboost_pred)

    # Print accuracies
    print("K-Nearest Neighbors (KNN) Accuracy:", knn_accuracy)
    print("Support Vector Machine (SVM) Accuracy:", svm_accuracy)
    print("Naive Bayes Accuracy:", nb_accuracy)
    print("Random Forest Accuracy:", rf_accuracy)
    print("AdaBoost Accuracy:", adaboost_accuracy)


if __name__ == "__main__":
    train_model()
