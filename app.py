import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

st.set_page_config(page_title="Laptop Price Predictor - Nigeria NG", layout="wide")
st.title("💻 Laptop Price Predictor - Nigeria NG")
st.write("Get estimated laptop price in ₦")

@st.cache_resource
def load_model():
    url = "https://huggingface.co/Noriz-Code/laptop-price-ng-model/resolve/main/best_laptop_price_model.pkl"
    response = requests.get(url)
    model = joblib.load(BytesIO(response.content))
    return model

# INPUTS - CHANGE THESE TO MATCH YOUR TRAINING COLUMNS
col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", ["Dell", "HP", "Lenovo", "Apple", "Asus", "Acer"])
    ram = st.selectbox("RAM GB", [4, 8, 16, 32])
    cpu = st.selectbox("CPU", ["Intel Core i3", "Intel Core i5", "Intel Core i7"])

with col2:
    storage = st.selectbox("Storage GB", [256, 512, 1024])
    gpu = st.selectbox("GPU", ["Intel", "Nvidia", "AMD"])
    os = st.selectbox("OS", ["Windows 10", "Windows 11"])

if st.button("Predict Price in NGN", type="primary"):
    with st.spinner("Loading model..."):
        model = load_model()
        
        # IMPORTANT: CHANGE COLUMN NAMES TO MATCH YOUR MODEL
        input_data = pd.DataFrame([[brand, ram, cpu, storage, gpu, os]],
                                  columns=["Brand", "Ram", "Cpu", "StorageGB", "Gpu", "Os"])
        
        prediction = model.predict(input_data)
        st.success(f"### Estimated Price: ₦{prediction[0]:,.2f}")
