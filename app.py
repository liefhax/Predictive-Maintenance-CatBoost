import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="centered")
st.title("Sistem Prediksi Kegagalan Mesin (CatBoost)")
st.write("Masukkan data indikator sensor di bawah ini untuk menganalisis risiko kerusakan mesin.")

@st.cache_resource
def load_my_model():
    model = CatBoostClassifier()
    model.load_model('model_catboost.cbm') # Pastikan file ini satu folder dengan app.py
    return model

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file 'model_catboost.cbm' sudah ada di folder yang sama. Error: {e}")

st.subheader("Indikator Sensor Mesin")

col1, col2 = st.columns(2)

with col1:
    type_input = st.selectbox("Tipe Kualitas Mesin (Type)", options=['L', 'M', 'H'], help="L = Low, M = Medium, H = High")
    air_temp = st.number_input("Air Temperature [K]", min_value=200.0, max_value=400.0, value=300.0)
    process_temp = st.number_input("Process Temperature [K]", min_value=200.0, max_value=400.0, value=310.0)

with col2:
    rotational_speed = st.number_input("Rotational Speed [rpm]", min_value=0, max_value=5000, value=1500)
    torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0)
    tool_wear = st.number_input("Tool Wear [min]", min_value=0, max_value=500, value=0)

if st.button("Analisis Kondisi Mesin", type="primary"):
    input_data = pd.DataFrame([{
        'Type': type_input,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rotational_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear
    }])
    
    # prediksi jenis kegagalan
    prediction = model.predict(input_data)
    result = prediction[0][0] if hasattr(prediction, 'shape') and len(prediction.shape) > 1 else prediction[0]
    
    st.markdown("---")
    st.subheader("Hasil Analisis Sistem:")
    
    if result == "No Failure" or result == 0:
        st.success(f"✅ **Mesin Aman:** Tidak terdeteksi indikasi kegagalan sistem.")
    else:
        st.error(f"⚠️ **Peringatan Bahaya:** Terdeteksi risiko kegagalan berjenis **[{result}]**! Segera lakukan maintenance.")