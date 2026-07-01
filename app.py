import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

model = tf.keras.models.load_model("flower_model.h5")

classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

st.title("🌸 Flower Image Classification")

uploaded_file = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    st.success(f"Predicted Flower: {predicted_class}")