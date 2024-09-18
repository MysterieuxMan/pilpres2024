import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nlp_id.lemmatizer import Lemmatizer
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load data preprocessing functions
def clean_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

lemmatizer = Lemmatizer()

def lemmatize_text(text):
    return lemmatizer.lemmatize(text)

def remove_numbers(text):
    return re.sub('[0-9]+', '', text)

def remove_stopwords(text):
    stop_words = set(stopwords.words('indonesian'))
    return ' '.join([word for word in nltk.word_tokenize(text) if word.lower() not in stop_words])

# Load trained model and vectorizer
tfidf_vect = TfidfVectorizer(max_features=5000)
naive_bayes = MultinomialNB()

# Load your training data
training_data = pd.read_csv('DataTweetFix - Copy.csv')

# Preprocess and vectorize training data
training_data['full_text'] = training_data['full_text'].apply(lambda x: x.lower())
training_data['full_text'] = training_data['full_text'].apply(clean_punctuation)
training_data['full_text'] = training_data['full_text'].apply(lemmatize_text)
training_data['full_text'] = training_data['full_text'].apply(remove_numbers)
training_data['full_text'] = training_data['full_text'].apply(remove_stopwords)

# Fit the vectorizer on training data
tfidf_vect.fit(training_data['full_text'])

# Vectorize training data
train_vectors = tfidf_vect.transform(training_data['full_text'])

# Fit the model on vectorized training data
naive_bayes.fit(train_vectors, training_data['text_weight_label'])

# Streamlit app
st.title("Sentimen Analisis")

# Get user input
user_input = st.text_area("Masukkan teks untuk dianalisis")

if st.button("Prediksi Sentimen"):
    # Preprocess user input
    processed_text = user_input.lower()
    processed_text = clean_punctuation(processed_text)
    processed_text = lemmatize_text(processed_text)
    processed_text = remove_numbers(processed_text)
    processed_text = remove_stopwords(processed_text)

    # Vectorize input
    input_vector = tfidf_vect.transform([processed_text])

    # Make prediction
    prediction = naive_bayes.predict(input_vector)
    sentiment = "Positif" if prediction[0] == 1 else "Negatif"

    st.write(f"Sentimen teks: {sentiment}")