import streamlit as st
import joblib
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import random

# Load the saved models
svm_model = joblib.load('svm.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')


# Create a text input for the user
def main():
    st.title("Sentiment Analysis Pemilu Presiden")


    daftar_kalimat = [
    "Pemilu kali ini memberikan harapan besar bagi masa depan bangsa yang lebih baik.",
    "Pilihlah pemimpin yang memiliki komitmen kuat untuk membangun negeri ini dengan integritas.",
    "Setiap suara yang diberikan dengan penuh tanggung jawab akan membawa perubahan positif untuk generasi mendatang."
    "Pemilu tahun ini penuh dengan janji-janji kosong yang sulit dipercaya.",
    "Saya rasa pemimpin yang terpilih nanti tidak akan membawa banyak perubahan bagi kita.",
    "Banyak orang yang apatis terhadap pemilu karena merasa suaranya tidak dihargai.",
    "Pemilu hanya formalitas tanpa ada niat nyata untuk memperbaiki kondisi negara."
    "Pemilu adalah bagian dari proses demokrasi yang harus dijalani setiap lima tahun sekali.",
    "Hari pemilihan berlangsung pada tanggal yang telah ditetapkan oleh KPU.",
    "Setiap warga negara yang terdaftar memiliki hak untuk memberikan suara di pemilu."
    ]

    if 'tes' not in st.session_state:
        st.session_state.tes = ""

    if st.button('➕ Create Sentiment Example'):
        kalimat = random.choice(daftar_kalimat)
        st.session_state.tes = kalimat
        st.experimental_rerun()

    user_input = st.text_area("Enter text for sentiment analysis:",
                            value=st.session_state.tes,
                            placeholder="Contoh text akan muncul di sini")
  
    if st.button("Analyze Sentiment"):
        if user_input:
            input_vector = tfidf_vectorizer.transform([user_input])
        
        # Make a prediction using the loaded SVM model
            prediction = svm_model.predict(input_vector)
            
            # Display the result
            st.write(f"The text is classified as: {prediction[0]}")
        else:
            st.write("Please enter some text to analyze.")

    # if st.button("Show Detailed Classification Report"):
    #     input_vector = Tfidf_vect.transform([user_input])
            
    #     prediction = model.predict(input_vector)
            
    #     proba = model.predict_proba(input_vector)[0]
    #     confidence = np.max(proba)
            
    #     st.write(f"Predicted Sentiment: {prediction[0]}")
    #     st.write(f"Confidence: {confidence:.2f}")



if __name__ == "__main__":
    main()