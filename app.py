import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("lstm_model.keras", compile=False)

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load max length
with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# Streamlit UI
st.title("Next Word Predictor")
st.write("Enter a sentence and predict the next word.")

text = st.text_input("Enter Text")

if st.button("Predict Next Word"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    
    else:
        # Convert text to sequence
        seq = tokenizer.texts_to_sequences([text])[0]

        # Padding
        padded = pad_sequences([seq], maxlen=max_len - 1, padding='pre')

        # Prediction
        pred = model.predict(padded, verbose=0)

        predicted_index = np.argmax(pred)

        # Find predicted word
        predicted_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                predicted_word = word
                break

        st.success(f"Predicted Next Word: {predicted_word}")
