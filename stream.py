import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import pandas as pd
import numpy as np
import random
from imblearn.over_sampling import SMOTE

@st.cache_data
def load_data():
    data = pd.read_csv('experiment.csv')

    data['full_text'] = data['full_text'].astype(str)
    return data

@st.cache_resource
def preprocess_and_train(data):
    X_train, X_test, y_train, y_test = train_test_split(data['full_text'], data['indoBert Classification'], test_size=0.2, random_state=42)

    Tfidf_vect = TfidfVectorizer(max_features=5000)
    X_train_Tfidf = Tfidf_vect.fit_transform(X_train)
    X_test_Tfidf = Tfidf_vect.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_Tfidf, y_train)

    model = joblib.load('SVM_exs.joblib')

    model.fit(X_train_resampled, y_train_resampled)

    return model, Tfidf_vect, X_test_Tfidf, y_test

def main():
    st.title("Sentiment Analysis Pemilu Presiden")

    data = load_data()

    model, Tfidf_vect, X_test_Tfidf, y_test = preprocess_and_train(data)

    train_accuracy = model.score(Tfidf_vect.transform(data['full_text']), data['indoBert Classification'])
    test_accuracy = model.score(X_test_Tfidf, y_test)

    daftar_kalimat = [
        "Pemilu kali ini memberikan harapan besar bagi masa depan bangsa yang lebih baik.",
        "Pilihlah pemimpin yang memiliki komitmen kuat untuk membangun negeri ini dengan integritas.",
        "Setiap suara yang diberikan dengan penuh tanggung jawab akan membawa perubahan positif untuk generasi mendatang.",
        "Pemilu tahun ini penuh dengan janji-janji kosong yang sulit dipercaya.",
        "Saya rasa pemimpin yang terpilih nanti tidak akan membawa banyak perubahan bagi kita.",
        "Banyak orang yang apatis terhadap pemilu karena merasa suaranya tidak dihargai.",
        "Pemilu hanya formalitas tanpa ada niat nyata untuk memperbaiki kondisi negara.",
        "Pemilu adalah bagian dari proses demokrasi yang harus dijalani setiap lima tahun sekali.",
        "Hari pemilihan berlangsung pada tanggal yang telah ditetapkan oleh KPU.",
        "Setiap warga negara yang terdaftar memiliki hak untuk memberikan suara di pemilu.",
        "Partisipasi aktif dalam pemilu adalah wujud nyata kepedulian kita terhadap nasib bangsa.",
        "Kita harus kritis dalam memilih pemimpin agar tidak tertipu oleh janji manis semata.",
        "Pemilu memberikan kesempatan bagi rakyat untuk menyuarakan aspirasinya.",
        "Banyak calon pemimpin yang hanya muncul saat pemilu tanpa kontribusi nyata sebelumnya.",
        "Semoga pemilu kali ini berjalan dengan jujur dan adil tanpa kecurangan.",
        "Pemilu seringkali dijadikan ajang untuk saling menjatuhkan lawan politik.",
        "Pemuda memiliki peran penting dalam menentukan arah masa depan melalui pemilu.",
        "Pengawasan ketat diperlukan untuk memastikan pemilu berjalan transparan.",
        "Kampanye yang sehat seharusnya berfokus pada program, bukan pada serangan pribadi.",
        "Mari gunakan hak pilih kita untuk memilih pemimpin yang benar-benar peduli pada rakyat.",
        "Kampanye pemilu kali ini dipenuhi dengan fitnah dan berita palsu.",
        "Banyak kandidat yang lebih mementingkan kepentingan pribadi daripada kepentingan rakyat.",
        "Saya tidak percaya bahwa pemilu akan membawa perubahan yang berarti.",
        "Korupsi masih merajalela meski pergantian pemimpin terus terjadi.",
        "Janji-janji politik hanya manis di awal tetapi pahit dalam realisasinya.",
         "Saya merasa suara rakyat tidak dihargai dalam pemilu kali ini.",
        "Pemilu hanya menguntungkan segelintir elit politik saja.",
        "Kecurangan dalam pemilu sudah menjadi rahasia umum yang dibiarkan.",
        "Kandidat yang ada tidak ada yang benar-benar mewakili kepentingan rakyat kecil.",
        "Perubahan yang dijanjikan hanyalah ilusi untuk meraih dukungan sementara.",
        "Pemilu kali ini bisa menjadi langkah awal menuju pemerintahan yang lebih transparan dan adil.",
        "Setiap warga negara memiliki kekuatan untuk menciptakan perubahan positif melalui hak pilihnya.",
        "Dengan memilih pemimpin yang tepat, kita bisa mewujudkan visi bangsa yang lebih sejahtera dan inklusif.",
        "Pemilu bukan hanya tentang memilih, tetapi juga tentang memperjuangkan masa depan yang lebih baik untuk semua.",
        "Semoga pemilu kali ini membawa pemimpin yang benar-benar peduli pada kemajuan sosial, ekonomi, dan kesejahteraan rakyat."
    ]

    if 'tes' not in st.session_state:
        st.session_state.tes = ""

    if st.button('➕ Create Sentiment Example'):
        kalimat = random.choice(daftar_kalimat)
        st.session_state.tes = kalimat
        st.rerun()

    user_input = st.text_area("Enter text for sentiment analysis:",
                            value=st.session_state.tes,
                            placeholder="Contoh text akan muncul di sini")
  
    if st.button("Analyze Sentiment"):
        if user_input:
            input_vector = Tfidf_vect.transform([user_input])
            
            prediction = model.predict(input_vector)
            
            proba = model._predict_proba_lr(input_vector)[0]
            # confidence = np.max(proba)

            # proba = model.predict_proba(input_vector)[0]
            confidence = np.max(proba)
            
            st.write(f"Predicted Sentiment: {prediction[0]}")
            st.write(f"Confidence: {confidence:.2f}")
        else:
            st.write("Please enter some text to analyze.")

    # if st.button("Show Detailed Classification Report"):
    #     input_vector = Tfidf_vect.transform([user_input])
            
    #     prediction = model.predict(input_vector)
            
    #     proba = model.predict_proba(input_vector)[0]
    #     confidence = np.max(proba)
            
    #     st.write(f"Predicted Sentiment: {prediction[0]}")
    #     st.write(f"Confidence: {confidence:.2f}")

    
    # if st.button("Accuracy Model"):     
    #     st.write(f"Model Training Accuracy: {train_accuracy:.2f}")
    #     st.write(f"Model Testing Accuracy: {test_accuracy:.2f}")

    #     y_pred = model.predict(X_test_Tfidf)
    #     report = classification_report(y_test, y_pred)
    #     st.text("Classification Report:")
    #     st.text(report)

        

if __name__ == "__main__":
    main()