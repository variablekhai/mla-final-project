import streamlit as st
import joblib
import pandas as pd
from datetime import date

# Load model and encoders
model = joblib.load('xgboost_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')  # This must match your training encoders

st.title("🚗 Traffic Pressure Prediction App")

# User inputs
vehicle_type = st.selectbox("Select Vehicle Type", options=['motokar', 'motokar_pelbagai_utiliti', 'jip', 'pick_up', 'window_van'])
vehicle_maker = st.selectbox("Select Vehicle Maker", options=['Perodua', 'Proton', 'Toyota'])
vehicle_model = st.selectbox("Select Vehicle Model", options=['Bezza', 'Saga', 'City', 'Vios', 'Hilux', 'Alza'])
fuel_type = st.selectbox("Select Fuel Type", options=['petrol', 'diesel', 'electric'])
state = st.selectbox("Select State", options=[
    'W.P. Kuala Lumpur', 'Selangor', 'Johor', 'Pulau Pinang', 'Sabah', 'Sarawak',
    'Kedah', 'Kelantan', 'Terengganu', 'Pahang', 'Negeri Sembilan', 'Melaka', 'Perak'
])
registration_date = st.date_input("Registration Date", value=date.today())

# Prediction trigger
if st.button("Predict Traffic Pressure 🔍"):
    # Prepare input
    input_data = pd.DataFrame({
        'type': [vehicle_type],
        'maker': [vehicle_maker],
        'model': [vehicle_model],
        'fuel': [fuel_type],
        'state': [state],
        'year': [registration_date.year],
        'month': [registration_date.month]
    })

    # Apply encoders
    for column in ['type', 'maker', 'model', 'fuel', 'state']:
        le = label_encoders[column]
        input_data[column] = le.transform(input_data[column])

    # Predict
    prediction = model.predict(input_data)

    # Output
    if prediction[0] == 1:
        st.success("🔺 Predicted Traffic Pressure: **High Traffic**")
    else:
        st.info("🟢 Predicted Traffic Pressure: **Low Traffic**")
