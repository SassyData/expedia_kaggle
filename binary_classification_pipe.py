import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import metrics
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
import pickle

def load_and_split_preprocessed_data(file_path = 'data/train_preprocessed.csv'):
    train_preprocessed = pd.read_csv(file_path)
    print(f"Pre-processed Data Loaded. \nShape: {train_preprocessed.shape}")

    # Randomly, split the data into test/training/validation sets
    x = train_preprocessed.drop(['prop_booking_bool'], axis=1)
    y = train_preprocessed['prop_booking_bool']
    x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.33, random_state=42)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # Check the balance of the splits on y
    print(y_test.mean(), y_train.mean())
    return x_train, x_test, y_train, y_test

def calc_variable_importance(x_train, y_train, x_test, n = 12, plot=True, export_file="models/top_feats.csv"):
    # Variable importance from Random Forest (could try PCA instead?)
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    feature_importances = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), x_train), reverse=True)
    feature_importances = pd.DataFrame(feature_importances[:20])
    print(f"Feature Importance : {feature_importances}")

    if plot:
        plt.rcParams['font.size'] = '28'
        plt.figure(figsize=(40, 20))
        plt.xticks(rotation=45)
        plt.bar(feature_importances[1], feature_importances[0])
        plt.xlabel('Variable', fontsize=30)
        plt.ylabel('Importance', fontsize=30)
        plt.show()

    # Take top 12 variables,
    top_feats = feature_importances.iloc[:n, 1].tolist()
    if export_file:
        pd.Series(top_feats).to_csv(export_file)

    x_train_min, x_test_min = x_train[top_feats], x_test[top_feats]
    return x_train_min, x_test_min

def resample_data(x_train, y_train):
    """Over and under sampling - using the SMOTE and Tomek links
    - SMOTE (Synthetic Minority Oversampling TEchnique) consists of synthesizing elements for the minority class
    - Tomek links are pairs of very close instances, but of opposite classes. Removing the instances of the majority
     class of each pair increases the space between the two classes"""
    smt = SMOTETomek(random_state=42)
    X_smt, y_smt = smt.fit_sample(x_train, y_train)

    # Replace training splits with resampled
    x_train_resampled, y_train_resampled = X_smt.copy(), y_smt.copy()
    print(x_train_resampled.shape, x_test.shape, y_train_resampled.shape, y_test.shape)
    # Check the balance of the splits on y
    print(f"Original y_test mean: {y_test.mean()}, Resampled y_train mean: {y_train_resampled.mean()}")

    return x_train_resampled, y_train_resampled

def run_model_get_results(x_train, y_train, x_test, model):
    # Run XGB
    # Fit
    model = model.fit(x_train, y_train)
    # Accuracy
    training_accuracy = model.score(x_train, y_train)
    print(f"Training accuracy: {training_accuracy}")

    # Predictions/probs on the test dataset
    predicted = pd.DataFrame(model.predict(x_test))
    probs = pd.DataFrame(model.predict_proba(x_test))

    # Store metrics
    accuracy = metrics.accuracy_score(y_test, predicted)
    roc_auc = metrics.roc_auc_score(y_test, probs[1])
    confus_matrix = metrics.confusion_matrix(y_test, predicted)
    classification_report = metrics.classification_report(y_test, predicted)
    precision = metrics.precision_score(y_test, predicted, pos_label=1)
    recall = metrics.recall_score(y_test, predicted, pos_label=1)
    f1 = metrics.f1_score(y_test, predicted, pos_label=1)

    print(f"CONFUSION MATRIX: \n{confus_matrix} \n\nCLASSIFICATION REPORT: \n{classification_report}")

    return accuracy, roc_auc, precision, recall, f1


def cv_evaluation(x_test, y_test, model):
    # Evaluate the model using 10-fold cross-validation
    cv_scores = cross_val_score(model, x_test, y_test, scoring='precision', cv=10)
    cv_mean = np.mean(cv_scores)
    print(f"Model Cross Validation Average: {cv_mean}")
    return cv_mean


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_and_split_preprocessed_data()

    # x_train_min, x_test_min = calc_variable_importance(x_train, y_train, x_test, n=12, plot=False, export_file=None)

    # x_train_resampled, y_train_resampled = resample_data(x_train_min, y_train)

    # Instantiate
    xgb_model = XGBClassifier(random_state=42)
    xgb_accuracy, xgb_roc_auc, xgb_precision, xgb_recall, xgb_f1 = run_model_get_results(x_train,
                                                                                         y_train,
                                                                                         x_test,
                                                                                         model=xgb_model)

    xgb_cv_mean = cv_evaluation(x_test, y_test, model=xgb_model)

    # write the models to disk
    pickle.dump(xgb_model, open("models/xgb_model_allvars_nosampling.sav", 'wb'))

    print(f"xgb_accuracy: {xgb_accuracy}, \nxgb_roc_auc: {xgb_roc_auc}, "
          f"\nxgb_precision: {xgb_precision}, "
          f"\nxgb_recall: {xgb_recall}, \nxgb_f1: {xgb_f1}")