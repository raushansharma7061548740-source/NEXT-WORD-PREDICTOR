import streamlit as st
import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("Next Word Predictor")
st.write("Enter text and predict the next word.")

@st.cache_resource
def load_lstm_model():
    if os.path.exists("lstm_model.keras"):
        return load_model("lstm_model.keras", compile=False)
    elif os.path.exists("lstm_model.h5"):
        return load_model("lstm_model.h5", compile=False)
    else:
        st.error("Model file not found. Please upload lstm_model.h5 or lstm_model.keras")
        st.stop()

@st.cache_resource
def load_files():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return tokenizer, max_len

model = load_lstm_model()
tokenizer, max_len = load_files()

text = st.text_input("Enter your text:")

if st.button("Predict Next Word"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        sequence = tokenizer.texts_to_sequences([text])[0]
        padded_sequence = pad_sequences([sequence], maxlen=max_len - 1, padding="pre")

        prediction = model.predict(padded_sequence, verbose=0)
        predicted_index = np.argmax(prediction)

        index_word = {index: word for word, index in tokenizer.word_index.items()}
        predicted_word = index_word.get(predicted_index, "Word not found")

        st.success(f"Predicted Next Word: {predicted_word}")
