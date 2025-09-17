import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -------- تحميل الموديل من Google Drive --------
MODEL_PATH = "alzheimer_model.h5"
FILE_ID = "1cb0L_Z1tPIaNfXwyk1D1Srm49btrfJD6"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)


model = load_model(MODEL_PATH)

# -------- واجهة Streamlit --------
st.title("🧠 Alzheimer MRI Classification")
st.write("Upload an MRI image to classify Alzheimer's stage.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)

    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0


    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)


    class_names = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

    st.success(f"### 🧾 Prediction: {class_names[class_idx]}")

#https://drive.google.com/file/d/1cb0L_Z1tPIaNfXwyk1D1Srm49btrfJD6/view?usp=sharing