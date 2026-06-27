# 🛍️ Smart Retail Intelligence Platform

## 🔗 Live Demo
**[Click here to view the live app](https://smart-retail-intelligence-platform-j2fyztyndi5rtxee7qzvhb.streamlit.app/)**

---

## 📌 Project Overview
End-to-end Machine Learning project for retail analytics covering customer segmentation, churn prediction, sales forecasting, and product image classification.

---

## 📊 Modules
| Module | Technique | Result |
|--------|-----------|--------|
| Customer Segmentation | RFM Analysis + KMeans | 4 segments identified |
| Churn Prediction | LogisticRegression, RandomForest, XGBoost | 79.61% ROC-AUC |
| Sales Forecasting | ARIMA, SARIMA, LSTM | SARIMA best (RMSE: £70,700) |
| Product Classifier | CNN (MobileNetV2) | 97% accuracy |

---

## 🛠️ Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow, Statsmodels, Streamlit, Matplotlib, Seaborn

---

## 📁 Dataset
Download from Kaggle: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
Place CSV in data/ folder after download.

---

## ⚙️ Setup
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Download dataset from Kaggle and place in data/ folder
4. Run: streamlit run app/Home.py

---

## 📈 Key Results
- Total Customers Analyzed: 5,878
- Total Revenue Analyzed: £1.77M
- Date Range: Dec 2009 - Dec 2011
- CNN Training Images: 18,557
- Data Leakage detected and fixed (time-based train/test split)
