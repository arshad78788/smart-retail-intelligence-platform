import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title='Sales Forecasting', layout='wide')
st.title('📈 Sales Forecasting')
st.markdown('---')

BASE = '/content/drive/MyDrive/Smart_Retail_Project'
weekly_sales = pd.read_csv(os.path.join(BASE, 'data/weekly_sales.csv'))
weekly_sales['Date'] = pd.to_datetime(weekly_sales['Date'])

st.subheader('📊 Weekly Sales Trend')
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(weekly_sales['Date'], weekly_sales['Revenue'],
        color='#4ECDC4', linewidth=1.5, marker='o', markersize=3)
ax.set_title('Weekly Revenue Trend (2009-2011)')
ax.set_xlabel('Date')
ax.set_ylabel('Revenue (£)')
plt.tight_layout()
st.pyplot(fig)

st.markdown('---')
st.subheader('🏆 Model Comparison Results')

col1, col2, col3 = st.columns(3)
with col1:
    st.metric('ARIMA RMSE', '£1,20,445', delta='Worst')
with col2:
    st.metric('SARIMA RMSE', '£70,700', delta='Best')
with col3:
    st.metric('LSTM RMSE', '£95,000', delta='2nd')

st.markdown('---')
st.subheader('📌 Key Findings')
st.markdown('- **SARIMA** outperformed LSTM on this dataset')
st.markdown('- LSTM needs more data to beat statistical models')
st.markdown('- Clear seasonality detected in weekly sales')
st.markdown('- Revenue peaks in **October-November** (holiday season)')

st.markdown('---')
st.subheader('📷 Forecast Charts')
col1, col2 = st.columns(2)
with col1:
    arima_img = os.path.join(BASE, 'reports/arima_sarima_forecast.png')
    st.image(arima_img, caption='ARIMA vs SARIMA Forecast', use_column_width=True)
with col2:
    lstm_img = os.path.join(BASE, 'reports/final_forecast_comparison.png')
    st.image(lstm_img, caption='ARIMA vs SARIMA vs LSTM', use_column_width=True)
