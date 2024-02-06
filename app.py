from flask import Flask, render_template, request, g
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import nltk
import string
import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')
import re
import sqlite3

app = Flask(__name__)
app.config['DATABASE'] = 'database.db'

svm_classifier = joblib.load('svm_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        # Initialize your database if needed

# Close the database connection when the application stops
app.teardown_appcontext(close_db)
# Initialize the database when the application starts
init_db()

def predict(input_data):
    def read_stopwords_from_file(stopwords_file):
        with open(stopwords_file, 'r', encoding='utf-8') as file:
            stopwords_list = [line.strip() for line in file]
        return set(stopwords_list)

    stopwords_file = 'stopwords.txt'

    stopwords_set = read_stopwords_from_file(stopwords_file)

    def get_top_words(news_text):
        stop_words = list(stopwords.words('bengali'))
        stop_words = set(stopwords.words('bengali'))
        stop_words = stop_words.union(stopwords_set)
        cleaned_words = [word.strip(string.punctuation + '।॥') for word in word_tokenize(news_text)]
        
        filtered_words = [word for word in cleaned_words if word.lower() not in stop_words and word.strip() and word not in ['‘', '’'] and not re.match(r'^[0-9০-৯]+$', word) and not re.match(r'^[a-zA-Z]+$', word)]
        
        filtered_words = [re.sub(r'[^ঀ-৿a-zA-Z0-9]', ' ', word) for word in cleaned_words if 
                        word.lower() not in stop_words and 
                        word.strip() and 
                        word not in ['‘', '’'] and 
                        not any(char.isdigit() or char in '০১২৩৪৫৬৭৮৯' for char in word)]
        
        fdist = FreqDist(filtered_words)
        
        top_words_and_freq = dict(sorted(fdist.items(), key=lambda item: item[1], reverse=True)[:10])
        
        headline = ' '.join([word for word, freq in top_words_and_freq.items()])
        
        return headline

    new_test_input = input_data

    new_test_tfidf = tfidf_vectorizer.transform([new_test_input])

    predicted_category = svm_classifier.predict(new_test_tfidf)

    category = predicted_category[0]

    query = f"SELECT * FROM data WHERE Category = '{predicted_category[0]}'"

    with app.app_context():
        db = get_db()
        filtered_rows = df = pd.read_sql_query(query, db)
        filtered_rows = filtered_rows.drop_duplicates()
        filtered_rows = filtered_rows.dropna()

    keywords = get_top_words(new_test_input)

    all_strings = pd.concat([filtered_rows['Headline'], pd.Series(keywords)], ignore_index=True)

    vectorizer = CountVectorizer(ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(all_strings)

    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    similar_strings_indices = cosine_similarities.argsort()[0][::-1][:5]

    similarity_scores = cosine_similarities[0, similar_strings_indices]

    similar_strings = filtered_rows['News'].iloc[similar_strings_indices].tolist()
    similar_category = filtered_rows['Category'].iloc[similar_strings_indices].tolist()
    similar_keywords = filtered_rows['Headline'].iloc[similar_strings_indices].tolist()
    output = list(zip(similar_strings, similarity_scores, similar_category, similar_keywords))
    return output, category, keywords

@app.route('/', methods=['GET', 'POST'])
def home():
    output = None

    if request.method == 'POST':
        input_data = request.form['input_data']
        output, predicted_category, predicted_keywords = predict(input_data)
        return render_template('index.html', input_data=input_data, output=output, predicted_category=predicted_category, predicted_keywords = predicted_keywords)

    return render_template('index.html', output=output)

@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        input_data = request.form['input_data']
        output, predicted_category, predicted_keywords = predict(input_data)
        return render_template('index.html', input_data=input_data, output=output, predicted_category=predicted_category, predicted_keywords = predicted_keywords)

if __name__ == '__main__':
    app.run(debug=True)
