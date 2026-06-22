import streamlit as st

st.set_page_config(page_title='Smart Retail Intelligence Platform', page_icon='🛍️', layout='wide')

st.title('🛍️ Smart Retail Intelligence Platform')
st.markdown('---')

st.markdown('## Welcome! 👋')
st.markdown('This platform provides end-to-end ML insights for retail businesses.')
st.markdown('---')

st.subheader('📊 Modules Available')
st.markdown('- 🎯 **Customer Segmentation** — RFM Analysis + KMeans Clustering')
st.markdown('- ⚠️ **Churn Prediction** — Predict which customers will leave')
st.markdown('- 📈 **Sales Forecasting** — ARIMA vs SARIMA vs LSTM')
st.markdown('- 👕 **Product Classifier** — CNN Image Classification (97% accuracy)')
st.markdown('---')

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric('Total Customers', '5,878')
with col2:
    st.metric('Total Revenue', '£1.77M')
with col3:
    st.metric('CNN Accuracy', '97%')
with col4:
    st.metric('Best Forecaster', 'SARIMA')
