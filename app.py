import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Laptop Price Predictor NG", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(r'C:\Users\userone\Desktop\models\best_laptop_price_model.pkl')

   model = joblib.load('best_laptop_price_model.pkl')

st.title("💻 Laptop Price Predictor - Nigeria NG")
st.write("Predict laptop price in NGN using GradientBoost_200")

st.sidebar.header("Enter Laptop Specs")

col1, col2 = st.sidebar.columns(2)
with col1:
    company = st.selectbox("Company", ['Dell', 'HP', 'Lenovo', 'Asus', 'Acer', 'Apple', 'MSI'])
    type_name = st.selectbox("Type", ['Notebook', 'Gaming', 'Ultrabook', '2 in 1 Convertible', 'Workstation'])
    ram = st.selectbox("RAM GB", [4, 8, 16, 32, 64])
    cpu_type = st.selectbox("CPU Type", ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'AMD'])
    cpu_brand = st.selectbox("CPU Brand", ['Intel', 'AMD'])
    cpu_speed = st.slider("CPU Speed GHz", 1.0, 4.5, 2.5, 0.1)
with col2:
    gpu = st.selectbox("GPU", ['Intel', 'Nvidia', 'AMD'])
    gpu_brand = st.selectbox("GPU Brand", ['Intel', 'Nvidia', 'AMD'])
    inches = st.slider("Screen Inches", 10.0, 18.0, 15.6, 0.1)
    screen_res = st.selectbox("Screen Resolution", ['1366x768', '1920x1080', '2560x1600', '3840x2160'])
    weight = st.slider("Weight KG", 0.8, 4.0, 2.0, 0.1)
    opsys = st.selectbox("OS", ['Windows 10', 'Windows 7', 'Mac OS', 'Linux', 'No OS'])

storage_type = st.sidebar.selectbox("Storage Type", ['SSD', 'HDD', 'Hybrid', 'Flash Storage'])
capacity_score = st.sidebar.number_input("Storage Capacity GB", 128, 2000, 512)
storage = st.sidebar.number_input("Storage GB", 128, 2000, 512)

storage_per_ram = capacity_score / ram

# ONLY ONE BUTTON
if st.button("Predict Price in NGN", type="primary", key="predict_btn"):
    
    input_data = {
        'Company': company, 'TypeName': type_name, 'ScreenResolution': screen_res,
        'Gpu': gpu, 'OpSys': opsys, 'Storage_Type': storage_type,
        'Cpu_Brand': cpu_brand, 'Cpu_Type': cpu_type, 'Gpu_Brand': gpu_brand,
        'Inches': inches, 'Ram': ram, 'Weight': weight, 'Storage': storage,
        'Storage_per_Ram': storage_per_ram, 'Capacity_Score': capacity_score,
        'Cpu_Speed': cpu_speed
    }
    
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    
    st.success(f"### Estimated Price: ₦{prediction:,.2f}")
    st.balloons()

st.caption("Model: GradientBoostingRegressor | All 16 features included")
