import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from PIL import Image

ps = PorterStemmer()

# 🔹 Cache NLTK downloads
@st.cache_resource
def load_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

load_nltk()

STOP_WORDS = set(stopwords.words('english'))

# 🔹 Load image safely
try:
    image = Image.open('images(2).jpg')
    st.image(image, caption='EMAIL')
except:
    st.warning("Image not found")

# 🔹 Text preprocessing
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    y = [i for i in tokens if i.isalnum()]
    y = [i for i in y if i not in STOP_WORDS and i not in string.punctuation]
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# 🔹 Load model & vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# 🔹 UI
st.title('Email Spam Classifier')

input_sms = st.text_input('Enter the Message')

option = st.selectbox("You Got Message From :-", ["Via Email", "Via SMS", "Other"])

if st.button('Click to Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message")
    else:
        transform_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transform_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")
