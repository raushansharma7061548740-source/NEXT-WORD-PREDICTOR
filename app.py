import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# Load Model and Files
# =========================

model = load_model("lstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Next Word Prediction using LSTM")
st.write("Enter a sentence and predict the next word.")

# User Input
input_text = st.text_input("Enter your text:")

# =========================
# Prediction Function
# =========================

def predict_next_word(model, tokenizer, text, max_len):

    # Convert text to sequence
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Padding
    token_list = pad_sequences(
        [token_list],
        maxlen=max_len - 1,
        padding='pre'
    )

    # Predict
    predicted = model.predict(token_list, verbose=0)

    # Get predicted index
    predicted_index = np.argmax(predicted)

    # Convert index to word
    output_word = ""

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            output_word = word
            break

    return output_word

# =========================
# Button
# =========================

if st.button("Predict Next Word"):

    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:

        next_word = predict_next_word(
            model,
            tokenizer,
            input_text,
            max_len
        )

        st.success(f"Predicted Next Word: {next_word}")

# =========================
# Footer
# =========================

st.markdown("---")
st.markdown("Built with Streamlit + TensorFlow")
