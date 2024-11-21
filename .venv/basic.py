import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import time
import os
import subprocess

def read_data(size=None):
    file_to_check = './DataSet/CleanedData/rejoined_Clean_file.csv'
    script_to_run = "./DataSet/CleanedData/RejoinCSV.py"
    if size is None:
        if not os.path.exists(file_to_check):
            try:
                subprocess.run(["python", script_to_run], check=True)
                print(f"Successfully ran {script_to_run}.")
            except subprocess.CalledProcessError as e:
                print(f"Error running {script_to_run}: {e}")
        else:
            print(f"{file_to_check} already exists. No action taken.")
        data = pd.read_csv(file_to_check, header=0)
        data = data.drop(data.columns[0], axis=1)
        data = data.dropna(subset=['clean_text'])
    elif size == 'small':
        data = pd.read_csv('cleaned_small_dataset.csv', header=0)
    return data

def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, data):
    start_time = time.time()
    if model_name == 'NaiveBayes':
        model.fit(X_train.toarray(), y_train)  # Convert sparse matrix for GaussianNB
        y_pred = model.predict(X_test.toarray())
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    file_name = f"{model_name}_incorrect_predictions.csv"
    print(f"\n--- {model_name} ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    report = classification_report(y_test, y_pred, target_names=['(0) 1.0, 2.0, or 3.0', '(1) 4.0 and 5.0'],
                                   zero_division=0)
    print(report)

    # Save incorrect predictions
    incorrect_indexes = np.where(y_test.to_numpy() != y_pred)[0]
    incorrect_indices = y_test.iloc[incorrect_indexes].index  # Get actual indices from y_test

    incorrect_df = pd.DataFrame({
        'original_index': incorrect_indices,
        'true_label': y_test.iloc[incorrect_indexes].values,
        'predicted_label': y_pred[incorrect_indexes],
        'actual_review_score': data.loc[incorrect_indices, 'overall'].values,  # Use loc to fetch by index
        'text': data.loc[incorrect_indices, 'clean_text'].values  # Use loc to fetch by index
    })
    incorrect_df.to_csv(file_name, index=False)
    print(f"Incorrect predictions saved to {file_name}")

    end_time = time.time()
    print(f'Execution time: {end_time - start_time:.4f} seconds')


# Main execution
data = read_data()
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(data['clean_text'])
X_train, X_test, y_train, y_test = train_test_split(X, data['rating'], test_size=0.2, random_state=42)

# Models
models = [
    (SVC(kernel='linear'), "SVM"),
    # (MultinomialNB(), "NaiveBayes"),
    (LogisticRegression(), "LogisticRegression")
]

for model, name in models:
    train_and_evaluate(model, name, X_train, X_test, y_train, y_test, data)
