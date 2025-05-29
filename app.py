import streamlit as st
import pickle
import re
import string
import numpy as np

# Load model and vectorizer
model = pickle.load(open("fake_news_predict.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Cleaning function (same as used during training)
def clean_text(text):
    text = text.lower()
    text = re.sub("\[.*?\]", " ", text)
    text = re.sub("\\W", " ", text)
    text = re.sub("https?://\S+|www\.\S+", " ", text)
    text = re.sub("<.*?>+", " ", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub("\n", " ", text)
    text = re.sub("\w*\d\w*", " ", text)
    return text

# Streamlit GUI
st.title("📰 Fake News Detection App")
st.markdown("Enter a news article or paragraph to check if it's **fake or real**.")

user_input = st.text_area("🗞️ Paste the news text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        cleaned = clean_text(user_input)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]

        if prediction == 1:
            st.success("✅ This news appears to be **Real**.")
        else:
            st.error("❌ This news appears to be **Fake**.")
