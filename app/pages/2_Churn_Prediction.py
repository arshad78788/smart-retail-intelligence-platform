import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title='Churn Prediction', layout='wide')
st.title('⚠️ Churn Prediction')
st.markdown('---')

BASE = '/content/drive/MyDrive/Smart_Retail_Project'
model = joblib.load(os.path.join(BASE, 'models/churn_model.pkl'))
scaler = joblib.load(os.path.join(BASE, 'models/churn_scaler.pkl'))

st.subheader('🔮 Predict Customer Churn')
st.markdown('Customer ki details enter karo:')

col1, col2, col3 = st.columns(3)

with col1:
    frequency = st.number_input('Frequency (Orders)', min_value=1, value=5)
    monetary = st.number_input('Monetary (Total Spend)', min_value=0.0, value=1000.0)

with col2:
    avg_order = st.number_input('Avg Order Value', min_value=0.0, value=200.0)
    total_items = st.number_input('Total Items', min_value=1, value=50)

with col3:
    unique_products = st.number_input('Unique Products', min_value=1, value=20)
    last_purchase = st.number_input('Days Since Last Purchase', min_value=1, value=90)

if st.button('🔮 Predict Churn', use_container_width=True):
    features = np.array([[frequency, monetary, avg_order,
                          total_items, unique_products, last_purchase]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    st.markdown('---')
    if prediction == 1:
        st.error(f'⚠️ HIGH CHURN RISK! Probability: {probability[1]*100:.1f}%')
        st.markdown('**Recommendation:** Offer discount or loyalty reward immediately!')
    else:
        st.success(f'✅ LOW CHURN RISK! Retention Probability: {probability[0]*100:.1f}%')
        st.markdown('**Recommendation:** Keep engaging with personalized offers!')
