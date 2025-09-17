# 🧠 Alzheimer MRI Classification (CNN + Streamlit)

A deep learning project for classifying **Alzheimer’s disease stages** from MRI images using a Convolutional Neural Network (CNN).  
The model is trained on the **Augmented Alzheimer MRI Dataset** from Kaggle, and deployed with a **Streamlit** web app for interactive predictions.

---

## 🌐 Live Demo
Try the deployed app here:  
👉 [Alzheimer MRI Classification App] (https://alzheimermriclassificationcnn-2wbeggyxfcdarsd7mvt4dn.streamlit.app/)

---

## 📂 Project Structure
- `task3_Alzheimer_Classification_(CNN).ipynb` → Model training and dataset preprocessing  
- `alzheimer_model.h5` → Trained CNN model (stored on Google Drive)  
- `app.py` → Streamlit app for predictions  
- `requirements.txt` → Dependencies list  

---

## 📊 Dataset
We used the **(https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset))** which contains MRI images categorized into 4 classes:

- 🟢 Non Demented  
- 🟡 Very Mild Demented  
- 🟠 Mild Demented  
- 🔴 Moderate Demented  

---

## 🏗️ Model Training
1. Downloaded dataset from Kaggle using `opendatasets`.  
2. Split into **train (80%)** and **validation (20%)** using `split-folders`.  
3. Built a CNN model with multiple Conv2D and MaxPooling layers, plus Dropout for regularization.  
4. Trained with **Adam optimizer** and **EarlyStopping**.  
5. Achieved competitive validation/test accuracy.  
6. Saved the final model as `alzheimer_model.h5`.  

---

## 📥 Pretrained Model
If you don’t want to retrain, you can directly download the trained model:  

👉 [Download Alzheimer Model (Google Drive)](https://drive.google.com/file/d/1cb0L_Z1tPIaNfXwyk1D1Srm49btrfJD6/view?usp=sharing)

You can then place it in your project root directory and run the Streamlit app.

---

## 🚀 Run Locally

### 1️⃣ Clone the repository
```bash
git clone  (https://github.com/MahmoudAttia111/Alzheimer_MRI_Classification_CNN).git
cd alzheimer-mri-classification
```
### 2️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
### 3️⃣ Run the Streamlit app
bash
Copy code
## streamlit run app.py
🖼️ Streamlit App Features
Upload an MRI image (jpg, png, jpeg).

The model predicts the Alzheimer’s stage in real-time.

Results are displayed interactively in the browser.

## 📈 Results
Validation Accuracy: ~ 94%

Test Accuracy (Original Dataset): ~ 96%

## 🛠️ Tech Stack
Python 3.10+

TensorFlow / Keras

Streamlit

NumPy, Pandas, Matplotlib

gdown (for downloading model from Google Drive)

