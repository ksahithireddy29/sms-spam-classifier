import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# SMS preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    stop_words = set(stopwords.words('english'))

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


st.title("📧 Email / SMS Spam Classifier")

message_type = st.radio(
    "Select Message Type",
    ["SMS", "Email"]
)

# Load correct model
if message_type == "SMS":
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
else:
    tfidf = pickle.load(open('email_vectorizer.pkl', 'rb'))
    model = pickle.load(open('email_model.pkl', 'rb'))

input_text = st.text_area("Enter your message")

if st.button("Predict"):

    if message_type == "SMS":
        processed_text = transform_text(input_text)
        vector_input = tfidf.transform([processed_text])

    else:
        # Email model was trained on raw text
        vector_input = tfidf.transform([input_text])

    result = model.predict(vector_input)[0]

    if hasattr(model, "predict_proba"):
        confidence = max(model.predict_proba(vector_input)[0]) * 100

    if result == 1:
        st.error("🚨 Spam")
    else:
        st.success("✅ Not Spam")

    if hasattr(model, "predict_proba"):
        st.write(f"**Confidence:** {confidence:.2f}%")