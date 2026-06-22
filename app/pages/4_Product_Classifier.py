import streamlit as st
import numpy as np
import os
import joblib
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title='Product Classifier', layout='wide')
st.title('👕 Product Image Classifier')
st.markdown('---')

BASE = '/content/drive/MyDrive/Smart_Retail_Project'

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(os.path.join(BASE, 'models/cnn_product_classifier.h5'))
    le = joblib.load(os.path.join(BASE, 'models/label_encoder.pkl'))
    return model, le

model, le = load_model()

st.subheader('🔮 Upload Product Image for Classification')
st.markdown('Supported categories: Apparel, Accessories, Footwear, Personal Care')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Uploaded Image')
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, width=300)

    with col2:
        st.subheader('Prediction')
        img_array = np.array(image.resize((64, 64))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = le.classes_[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        st.success(f'Predicted Category: **{predicted_class}**')
        st.metric('Confidence', f'{confidence:.1f}%')

        st.markdown('---')
        st.subheader('All Probabilities')
        for i, cls in enumerate(le.classes_):
            prob = predictions[0][i] * 100
            st.progress(int(prob), text=f'{cls}: {prob:.1f}%')

st.markdown('---')
st.subheader('📊 Model Performance')
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Overall Accuracy', '97%')
with col2:
    st.metric('Model', 'MobileNetV2')
with col3:
    st.metric('Training Images', '18,557')

conf_img = os.path.join(BASE, 'reports/cnn_confusion_matrix.png')
st.image(conf_img, caption='Confusion Matrix', use_column_width=True)
