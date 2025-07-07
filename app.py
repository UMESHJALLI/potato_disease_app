import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Test: Potato Disease Detector")

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded image", use_column_width=True)

    image = Image.open(uploaded_file).resize((256, 256))
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    st.write("✅ Image preprocessed successfully.")

    # Dummy model prediction
    fake_prediction = [0.1, 0.2, 0.7]
    class_names = ['Early Blight', 'Late Blight', 'Healthy']
    predicted_class = class_names[np.argmax(fake_prediction)]

    st.success(f"Prediction: {predicted_class}")