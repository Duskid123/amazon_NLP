import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import time

def read_data(size=None):
    if size == None:
        data = pd.read_csv('./DataSet/cleaned_dataset.csv', )
    if size == 'small':
        data = pd.read_json('cleaned_small_dataset.json', lines=True)
    return data

"""
    X_train is the cleaned text that is used to train the model
    X_test is the cleaned text that is used to test the model
    y_train is the rating for the clean text provided and is provided to the model
    y_test is the actual rating the X_train is corrilated to

"""

def useSVM(X_train, X_test, y_train, y_test):
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    return y_pred, 'SVM_incorrect_predictions.csv'

def useNaiveBayes(X_train, X_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(X_train.toarray(), y_train)  # Convert sparse matrix for GaussianNB
    y_pred = gnb.predict(X_test.toarray())
    return y_pred, 'NB_incorrect_predictions.csv'

def useLogisticRegression(X_train, X_test, y_train, y_test):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    return y_pred, 'LR_incorrect_predictions.csv'


def Analysis(y_test, y_pred, data, fileName):
    # Evaluate the model's performance
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    report = classification_report(y_test, y_pred, target_names=['(0) 1.0, 2.0, or 3.0', '(1) 4.0 and 5.0'],
                                   zero_division=0)
    print(report)

    # Identify incorrect predictions
    incorrect_indexes = np.where(y_test.to_numpy() != y_pred)[0]

    # Get the original indices for the incorrect predictions
    original_indices = y_test.iloc[incorrect_indexes].index

    # Create a DataFrame for incorrect predictions, including original index
    incorrect_df = pd.DataFrame({
        'original_index': original_indices,
        'true_label': y_test.iloc[incorrect_indexes].values,
        'predicted_label': y_pred[incorrect_indexes],
        'actual_review_score': data.iloc[original_indices]['overall'].values,  # Check correct column
        'text': data.iloc[original_indices]['clean_text'].values  # Check correct column
    })

    # Display or save the DataFrame of incorrect predictions
    incorrect_df.to_csv('big'+fileName, index=False)
    print(incorrect_df.head())  # Print the first few rows for a quick check


data = read_data()
# This is to use TF-IDF to create feature extractions
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(data['clean_text'])


X_train, X_test, y_train, y_test = train_test_split(X, data['rating'], test_size=0.2, random_state=42)

start_time = time.time()

# SVM Analysis
SVMy_pred, svm_file = useSVM(X_train, X_test, y_train, y_test)
Analysis(y_test, SVMy_pred, data, svm_file)
end_time = time.time()

print(f'Execution time: {end_time - start_time:,.4f} seconds')

start_time = time.time()

# Naive Bayes Analysis
NBy_pred, nb_file = useNaiveBayes(X_train, X_test, y_train, y_test)
Analysis(y_test, NBy_pred, data, nb_file)

end_time = time.time()

print(f'Execution time: {end_time - start_time:,.4f} seconds')


start_time = time.time()

# Naive Bayes Analysis
LRy_pred, lr_file = useNaiveBayes(X_train, X_test, y_train, y_test)
Analysis(y_test, LRy_pred, data, lr_file)

end_time = time.time()

print(f'Execution time: {end_time - start_time:,.4f} seconds')