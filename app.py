import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

@st.cache_resource
def load_keras_assets():
    # Load the native .keras formatted model
    model = tf.keras.models.load_model('./saved_keras_model/spam_keras_model.keras')
    with open('./saved_keras_model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer, model

tokenizer, model = load_keras_assets()

# Streamlit App UI
st.title("✉ Keras Spam Classification App")
st.write("This app uses a Keras neural network to predict if a message is Spam or Ham.")

message = st.text_area("Enter your message below:", height=150)

if st.button("Classify Message"):
    if message.strip() == "":
        st.warning("Please enter a valid message.")
    else:
        # Preprocess input message
        seq = tokenizer.texts_to_sequences([message])
        padded = pad_sequences(seq, maxlen=128, padding='post', truncating='post')
        
        # Predict probability
        prob = model.predict(padded)[0][0]
        prediction = 1 if prob >= 0.5 else 0
        
        label_map = {0: 'Ham (Not Spam)', 1: 'Spam'}
        label = label_map[prediction]
        confidence = prob * 100 if prediction == 1 else (1 - prob) * 100
        
        if prediction == 1:
            st.error(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
