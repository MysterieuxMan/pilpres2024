import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import pandas as pd
import random
import tweepy
import re
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE

<<<<<<< HEAD
# Bearer token untuk API Twitter
=======
>>>>>>> 8ff2f1b7883b712f60014a1ef377255eda789f02
bearer_token = "AAAAAAAAAAAAAAAAAAAAAAthwAEAAAAAOYyLVEa1W5j%2FxoJKgerIXEasq1k%3D2MCSvMpAjgdqpNGg7dVAkOv7R41rBqznrCcwF9Q7WOZHgvYMtb"

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

    model = joblib.load('svc.joblib')

    if isinstance(model, LinearSVC):
        model.set_params(dual=False)

    model.fit(X_train_resampled, y_train_resampled)

    return model, Tfidf_vect, X_test_Tfidf, y_test

def read_text_file(file):
    try:
        return file.getvalue().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def analyze_text(text, model, Tfidf_vect):
    input_vector = Tfidf_vect.transform([text])
    prediction = model.predict(input_vector)
    return prediction[0]

def extract_tweet_id(url):
    patterns = [
        r'twitter\.com/\w+/status/(\d+)',
        r'x\.com/\w+/status/(\d+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_tweet_content(url):
    client = tweepy.Client(bearer_token=bearer_token)
    tweet_id = extract_tweet_id(url)
    
    if not tweet_id:
        return None
    try:
        tweet = client.get_tweet(tweet_id, tweet_fields=["created_at", "public_metrics"], expansions=["author_id"], user_fields=["username", "name"])
        if tweet and tweet.data:
            return tweet.data["text"]
        else:
            return None
    except tweepy.TweepyException as e:
        st.error(f"Error retrieving tweet: {str(e)}")
        return None

def analyze_text(text, model, Tfidf_vect):
    input_vector = Tfidf_vect.transform([text])
    prediction = model.predict(input_vector)
    return prediction[0]

def main():
    st.title("Sentimen Analisis")

    data = load_data()
    model, Tfidf_vect, X_test_Tfidf, y_test = preprocess_and_train(data)

    st.header("Upload File")
    uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
    
    if uploaded_file is not None:
        text_content = read_text_file(uploaded_file)
        if text_content:
            st.text_area("File Content:", value=text_content, height=200, disabled=True)
            
            if st.button("Analyze File Content"):
                sentiment = analyze_text(text_content, model, Tfidf_vect)
                st.write(f"Predicted Sentiment for File Content: {sentiment}")

    st.markdown("---")
    st.header("Manual Text Input")

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


    user_input = st.text_area("Enter text or tweet link for sentiment analysis:",
                            value=st.session_state.tes,
                            placeholder="Contoh text akan muncul di sini")
  
    if st.button("Analyze Text"):
        if user_input:
            if re.match(r'^(https?://)?(www\.)?(twitter|x)\.com/', user_input):
                tweet_content = get_tweet_content(user_input)
                if tweet_content:
                    st.write(f"**Tweet Content:**")
                    st.write(tweet_content)
                    prediction = analyze_text(tweet_content, model, Tfidf_vect)
                    st.write(f"**Predicted Sentiment:** {prediction}")
                else:
                    st.write("Could not retrieve tweet content. Please check the URL.")
            else:
                prediction = analyze_text(user_input, model, Tfidf_vect)
                st.write(f"**Predicted Sentiment:** {prediction}")
        else:
            st.write("Please enter some text or a tweet link to analyze.")

if __name__ == "__main__":
    main()
