import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Laptop Price Predictor - Nigeria NG", layout="wide")
st.title("💻 Laptop Price Predictor - Nigeria NG")
st.write("Fill the specs below and get an estimated price in ₦")

# ALL YOUR INPUTS HERE
col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", ["Dell", "HP", "Lenovo", "Apple", "Asus", "Acer"])
    type_name = st.selectbox("Type", ["Notebook", "Gaming", "Ultrabook", "2 in 1 Convertible"])
    ram = st.selectbox("RAM", [4, 8, 16, 32, 64])
    cpu = st.selectbox("CPU", ["Intel Core i3", "Intel Core i5", "Intel Core i7", "AMD Ryzen 5", "AMD Ryzen 7"])

with col2:
    gpu = st.selectbox("GPU", ["Intel", "Nvidia", "AMD"])
    screen_size = st.number_input("Screen Size Inch", 11.0, 18.0, 15.6)
    storage = st.selectbox("Storage GB", [256, 512, 1024, 2048])
    os = st.selectbox("OS", ["Windows 10", "Windows 11", "MacOS", "Linux"])
    weight = st.number_input("Weight KG", 0.5, 5.0, 2.0)

# THE MAGIC: LOAD MODEL ONLY WHEN BUTTON IS CLICKED
if st.button("Predict Price in NGN", type="primary"):
    with st.spinner("Loading model and predicting..."):
        model = joblib.load("best_laptop_price_model.pkl") # <-- loads here, not at start
        
        # YOU NEED TO CHANGE THIS PART TO MATCH YOUR TRAINING COLUMNS
        input_data = pd.DataFrame([[brand, type_name, ram, cpu, gpu, screen_size, storage, os, weight]],
                                  columns=["Brand", "TypeName", "Ram", "Cpu", "Gpu", "ScreenSizeInch", "StorageGB", "Os", "Weight"])
        
        prediction = model.predict(input_data)
        st.success(f"### Estimated Price: ₦{prediction[0]:,.2f}")
