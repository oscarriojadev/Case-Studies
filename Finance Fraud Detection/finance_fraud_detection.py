# finance_fraud_detection.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Loaded data.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data, target_column):
    """
    Preprocess the data by splitting into features and target, and then scaling the features.

    Parameters:
    data (pd.DataFrame): Input data.
    target_column (str): Name of the target column.

    Returns:
    tuple: Features (X) and target (y) arrays.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.

    Parameters:
    X (array): Features.
    y (array): Target.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Seed for the random number generator.

    Returns:
    tuple: Training and testing sets (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train a Random Forest model.

    Parameters:
    X_train (array): Training features.
    y_train (array): Training target.

    Returns:
    RandomForestClassifier: Trained Random Forest model.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using accuracy, precision, recall, and F1 score.

    Parameters:
    model: Trained model.
    X_test (array): Testing features.
    y_test (array): Testing target.

    Returns:
    tuple: Accuracy, precision, recall, and F1 scores.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

def plot_confusion_matrix(y_test, y_pred):
    """
    Plot the confusion matrix.

    Parameters:
    y_test (array): Actual values.
    y_pred (array): Predicted values.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance from the trained model.

    Parameters:
    model: Trained model.
    feature_names (list): List of feature names.
    """
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance')
    plt.show()

def main():
    # Example usage
    file_path = 'path_to_your_data.csv'
    target_column = 'Fraud'

    data = load_data(file_path)
    X, y = preprocess_data(data, target_column)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)

    feature_names = data.drop(columns=[target_column]).columns
    plot_feature_importance(model, feature_names)

    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
