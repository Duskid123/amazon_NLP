import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import time
import datetime
import os
import subprocess


def read_data(size=None):
    file_to_check = './DataSet/CleanedData/rejoined_Clean_file.csv'
    script_to_run = "./DataSet/CleanedData/RejoinCSV.py"
    if size == None:
        if not os.path.exists('./DataSet/CleanedData/rejoined_Clean_file.csv'):
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
    if size == 'small':
        data = pd.read_csv('./Dataset/cleaned_dataset1000.csv', header=0)
    return data

data = read_data()

# Function to create a unique directory
def create_unique_directory(base_name):
    directory = base_name
    counter = 1
    while os.path.exists(directory):
        directory = f"{base_name}_{counter}"
        counter += 1
    os.makedirs(directory)

    return directory

# This is to use TF-IDF to create feature extractions
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(data['clean_text'])

def getScore(model, X_train, X_test, y_train, y_test, data, filename, directory):
    # has the provided model run
    model.fit(X_train, y_train)

    # saves the prediced response with the
    y_pred = model.predict(X_test)
    # this is to be able to track the f1 score, accuracy, precision, recall for the specific fold and model
    save_message = ''

    accuracy = f'Accuracy: {accuracy_score(y_test, y_pred)}'

    print(accuracy)
    save_message += accuracy

    # this is a report to see how well the model did
    report = classification_report(y_test, y_pred, target_names=['(0) 1.0, 2.0, or 3.0', '(1) 4.0 and 5.0'],
                                       zero_division=0)
    print(report)
    save_message += '\n' + report

    # Identify incorrect predictions
    incorrect_indexes = np.where(y_test.to_numpy() != y_pred)[0]

    # this is so that I can see what the incorrectly predicted text was.
    original_indices = y_test.iloc[incorrect_indexes].index

    # Create a DataFrame for incorrect predictions, including original index
    incorrect_df = pd.DataFrame({
        'original_index': original_indices,
        'true_label': y_test.iloc[incorrect_indexes].values,
        'predicted_label': y_pred[incorrect_indexes],
        'actual_review_score': data.iloc[original_indices]['overall'].values,
        'text': data.iloc[original_indices]['clean_text'].values
    })

    a = open(directory+'TXT/'+filename+'.txt', 'w')
    a.write(save_message)
    a.close()

    # Display or save the DataFrame of incorrect predictions
    incorrect_df.to_csv(directory+'CSV/'+filename+'.csv', index=False)
    print(incorrect_df.head())  # Print the first few rows for a quick check



folds = StratifiedKFold(n_splits=5)

i = 1
data = data.reset_index(drop=True)

directory = create_unique_directory(f'KfoldAnalysis{i}')

fold_dir = os.path.join(directory, f"SVM")
os.makedirs(fold_dir, exist_ok=True)
fold_dir2 = os.path.join(fold_dir, f"TXT")
os.makedirs(fold_dir2, exist_ok=True)
fold_dir2 = os.path.join(fold_dir, f"CSV")
os.makedirs(fold_dir2, exist_ok=True)

fold_dir = os.path.join(directory, f"LR")
os.makedirs(fold_dir, exist_ok=True)
fold_dir2 = os.path.join(fold_dir, f"TXT")
os.makedirs(fold_dir2, exist_ok=True)
fold_dir2 = os.path.join(fold_dir, f"CSV")
os.makedirs(fold_dir2, exist_ok=True)

for train_index, test_index in folds.split(X, data['rating']):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], \
        data['rating'][train_index], data['rating'][test_index]
    print('Starting the ' + str(i) + 'th Lr fold split...')
    getScore(LogisticRegression(), X_train, X_test, y_train, y_test, data,
             filename='bigLRIncorrect'+str(i), directory=directory+'/LR/')
    print('Starting the ' + str(i) + 'th svm fold split...')
    getScore(SVC(), X_train, X_test, y_train, y_test, data, filename='bigSVCIncorrect'+str(i),
             directory=directory+'/SVM/')
    # print(getScore(GaussianNB(), X_train.toarray(), X_test, y_train, y_test))
    i+= 1
