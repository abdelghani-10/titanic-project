import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def main():
    # 1. LOAD DATA
    df = sns.load_dataset('titanic')

    
    df = df.drop(['deck', 'embark_town', 'alive', 'who'], axis=1)

    # 2. DATA CLEANING
    df['age'].fillna(df['age'].median(), inplace=True)
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
    df['fare'].fillna(df['fare'].median(), inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # 3. FEATURE ENGINEERING
    df['FamilySize'] = df['sibsp'] + df['parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['name'].str.extract(' ([A-Za-z]+)\.')

    # 4. SELECT FEATURES
    X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
            'embarked', 'FamilySize', 'IsAlone', 'Title']]
    y = df['survived']

    # 5. ENCODING + SCALING USING PIPELINE
    numeric_features = ['age', 'sibsp', 'parch', 'fare', 'FamilySize']
    categorical_features = ['pclass', 'sex', 'embarked', 'IsAlone', 'Title']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # 6. TRAIN / TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # 7. MODELS
    log_reg_clf = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', LogisticRegression(max_iter=500))
    ])

    knn_clf = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', KNeighborsClassifier(n_neighbors=5))
    ])

    # 8. TRAIN MODELS
    log_reg_clf.fit(X_train, y_train)
    knn_clf.fit(X_train, y_train)

    # 9. EVALUATION
    y_pred_lr = log_reg_clf.predict(X_test)
    print("=== Logistic Regression ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_lr))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
    print("Classification Report:\n", classification_report(y_test, y_pred_lr))

    y_pred_knn = knn_clf.predict(X_test)
    print("\n=== KNN ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_knn))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
    print("Classification Report:\n", classification_report(y_test, y_pred_knn))


if __name__ == "__main__":
    main()
