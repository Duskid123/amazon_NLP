import pandas as pd  # For data manipulation
import re  # For regular expressions
from nltk.tokenize import word_tokenize  # To split text into words
from nltk.stem import WordNetLemmatizer  # To lemmatize words
from nltk.corpus import stopwords
import time, os, subprocess


# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
size = 100

# Function to clean the text
def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()
    # Remove punctuation, numbers, and special characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize and lemmatize
    # words = [lemmatizer.lemmatize(word) for word in word_tokenize(text)]

    new_filtered_words = [
        word for word in word_tokenize(text) if word.lower() not in stopwords.words('english')]
    return ' '.join(new_filtered_words)

# Load and preprocess data
try:
    start_time = time.time()

    # if you wanted to clean all of the data just don't use sample
    file_to_check = './DataSet/unCleanedData/rejoined_fileUncleaned.csv'
    script_to_run = "./DataSet/unCleanedData/RejoinCSV.py"

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
    # data = pd.read_csv('./DataSet/unCleanedData/rejoined_fileUncleaned.csv')
except ValueError:
    print("Error loading JSON file. Please check the file path and format.")
else:
    # Clean text and generate binary rating
    data['clean_text'] = data['reviewText'].apply(clean_text)
    data['rating'] = (data['overall'] > 3).astype(int)  # 1 if higher than 3, 0 otherwise

    # Save the cleaned dataset to JSON
    data = data.dropna(subset=['clean_text'])
    data.to_csv('./Dataset/cleaned_dataset.csv')
    print(f"Data cleaning complete. Saved to 'cleaned_dataset.csv'.")
    end_time = time.time()

    print(f'Execution time: {end_time - start_time:,.4f} seconds')
