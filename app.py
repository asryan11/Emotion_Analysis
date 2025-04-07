import streamlit as st
import joblib
import re
from bs4 import BeautifulSoup
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load('ed.pkl')
vectorizer = joblib.load('vect.pkl')

# Preprocessing functions
def noiseremoval_text(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    text = re.sub('\[[^]]*\]', '', text)
    return text

def stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

def preprocess(text):
    text = noiseremoval_text(text)
    text = stemmer(text)
    return text

# Streamlit UI
st.set_page_config(page_title="Emotion Classifier", layout="centered")
st.title("🧠 Emotion Classifier")
st.write("Enter a sentence, and I'll predict the emotion!")

user_input = st.text_area("Your text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        processed = preprocess(user_input)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        st.success(f"**Predicted Emotion:** {prediction}")
