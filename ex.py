import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import joblib

# Load data (assuming you have a CSV file with 'full_text' and 'indoBert Classification' columns)
@st.cache_data
def load_data():
    data = pd.read_csv('experiment.csv')
    return data

# Preprocess and train model
@st.cache_resource
def train_model(data):
    X = data['full_text']
    y = data['indoBert Classification']

    Tfidf_vect = TfidfVectorizer(max_features=5000)
    X_train_Tfidf = Tfidf_vect.fit_transform(X)

    smote = SMOTE()
    tfidf_vector, y = smote.fit_resample(X_train_Tfidf, y)

    X_train, X_test, y_train, y_test = train_test_split(tfidf_vector, y, test_size=0.3, random_state=42)

    svm_model = SVC()

    param_grids = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    grid_search = GridSearchCV(svm_model, param_grids, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model_svm_grid = grid_search.best_estimator_

    return Tfidf_vect, best_model_svm_grid

# Streamlit app
def main():
    st.title('Text Classification with SVM')

    # Load data and train model
    data = load_data()
    Tfidf_vect, model = train_model(data)

    # User input
    user_input = st.text_area("Enter the text you want to classify:", "")

    if st.button('Classify'):
        if user_input:
            # Preprocess input
            user_input_tfidf = Tfidf_vect.transform([user_input])

            # Make prediction
            prediction = model.predict(user_input_tfidf)

            # Display result
            st.subheader('Classification Result:')
            st.write(f"The text is classified as: {prediction[0]}")
        else:
            st.warning("Please enter some text to classify.")

    # Option to display model details
    if st.checkbox('Show Model Details'):
        st.subheader('Model Information:')
        st.write(f"Best Parameters: {model.get_params()}")

if __name__ == '__main__':
    main()