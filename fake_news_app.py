import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

stop = set(stopwords.words('english'))
lem = WordNetLemmatizer()

# Load model + TF-IDF vectorizer
model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = " ".join(w for w in text.split() if w not in stop)
    text = " ".join(lem.lemmatize(w) for w in text.split())
    return text

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection App")
st.write("Enter a news headline or article, and the model will tell you whether it's REAL or FAKE.")

user_input = st.text_area("Enter news text below:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Bruh, type something first 😭")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("✅ REAL NEWS — looks trustworthy.")
        else:
            st.error("🚨 FAKE NEWS — looks suspicious!")