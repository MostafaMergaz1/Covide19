
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

model_path = r"C:\Users\ALMot7da\Documents\covid_19_model.h5"
model = load_model(model_path) 

class_map = {0: "COVID", 1: "Normal", 2: "Pneumonia"}

st.title("COVID-19 Detection from Chest X-ray")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error("Error loading image. Please upload a valid image file.")
        st.stop()
    
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Resize image to model input size
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0]  # prediction shape (3,)
    predicted_class = np.argmax(prediction)
    predicted_label = class_map[predicted_class]

    # Display main prediction
    st.success(f"**Prediction:** {predicted_label}")

    # Display probabilities for all classes
    st.subheader("Prediction Confidence")
    for i, label in class_map.items():
        st.write(f"{label}: {prediction[i]*100:.2f}%")

else:
    st.info("Please upload an image file to get prediction.")

