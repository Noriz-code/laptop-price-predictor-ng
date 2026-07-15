import streamlit as st
import joblib
import requests
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Laptop Price Predictor - Nigeria NG", layout="wide")
st.title("💻 Laptop Price Predictor - Nigeria NG")

@st.cache_resource
def load_model():
    url = "https://huggingface.co/Noriz-Code/laptop-price-ng-model/resolve/main/best_laptop_price_model.pkl"
    response = requests.get(url)
    model = joblib.load(BytesIO(response.content))
    return model

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
    model = load_model()
    
    # CHANGE THIS LIST TO MATCH THE ORDER OF COLUMNS YOU USED TO TRAIN
    input_list = [[brand, ram, cpu, storage, gpu, os]]
    prediction = model.predict(input_list)
    st.success(f"### Estimated Price: ₦{prediction[0]:,.2f}")
