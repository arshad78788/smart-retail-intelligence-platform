import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title='Customer Segmentation', layout='wide')
st.title('🎯 Customer Segmentation')
st.markdown('---')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rfm = pd.read_csv(os.path.join(BASE, 'data/rfm_clustered.csv'))

st.subheader('📊 Customer Segments Overview')
col1, col2, col3, col4 = st.columns(4)
segments = rfm['Segment'].value_counts()

with col1:
    st.metric('Champions', int(segments.get('Champions', 0)))
with col2:
    st.metric('At Risk', int(segments.get('At Risk', 0)))
with col3:
    st.metric('Lost Customers', int(segments.get('Lost Customers', 0)))
with col4:
    st.metric('New/Promising', int(segments.get('New/Promising', 0)))

st.markdown('---')
col1, col2 = st.columns(2)

with col1:
    st.subheader('Segment Distribution')
    fig, ax = plt.subplots(figsize=(6,6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    ax.pie(segments.values, labels=segments.index,
           autopct='%1.1f%%', colors=colors, startangle=90)
    st.pyplot(fig)

with col2:
    st.subheader('RFM Summary by Segment')
    summary = rfm.groupby('Segment').agg(
        Recency=('Recency', 'mean'),
        Frequency=('Frequency', 'mean'),
        Monetary=('Monetary', 'mean'),
        Count=('Customer ID', 'count')
    ).round(2)
    st.dataframe(summary, use_container_width=True)

st.markdown('---')
st.subheader('🔍 Customer Search')
customer_id = st.number_input('Customer ID enter karo:', min_value=0, step=1)
if st.button('Search'):
    result = rfm[rfm['Customer ID'] == customer_id]
    if len(result) > 0:
        st.success('Customer found!')
        st.dataframe(result)
    else:
        st.error('Customer not found!')
