import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model('potato_disease_model.h5')

# Class names (adjust if your model uses different labels)
class_names = ['Early Blight', 'Late Blight', 'Healthy']

# Title
st.title("🌿 Potato Leaf Disease Detection")

# File uploader
uploaded_file = st.file_uploader("Upload a potato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file).resize((256, 256))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    result = class_names[np.argmax(prediction)]

    st.success(f"Prediction: *{result}* 🌱")