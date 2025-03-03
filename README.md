# Zembo-Data-Science

# **Predictive Modeling for Battery Swap Optimization**

## **Overview**
This project aims to optimize **battery swap operations** for electric vehicles by leveraging **predictive modeling** and **anomaly detection**. The goal is to forecast swap demand and detect battery anomalies using **time-series forecasting techniques** such as **XGBoost and LSTM**.

## **Project Workflow**
1. **Data Collection & Preprocessing:** Battery data and swap event logs were collected and cleaned.
2. **Exploratory Data Analysis (EDA):** Statistical insights were derived to understand patterns and correlations.
3. **Feature Engineering:** Additional predictive features such as **battery usage rate**, **swap frequency**, and **time lags** were created.
4. **Predictive Modeling:** XGBoost and LSTM models were trained to forecast swap demand.
5. **Anomaly Detection:** Isolation Forest was used to identify faulty battery behaviors.
6. **Evaluation & Business Insights:** Model performance was assessed, and actionable recommendations were provided.

---
## **Justification of Tools & Techniques**

### **1. Libraries & Frameworks Used**
- **Pandas & NumPy:** For data manipulation and preprocessing.
- **Matplotlib & Seaborn:** For visualizing trends, distributions, and correlations.
- **Scikit-Learn & XGBoost:** For feature engineering and machine learning.
- **TensorFlow/Keras:** For deep learning-based forecasting with LSTM.
- **Google Colab & Google Drive:** For cloud-based execution and data storage.

### **2. Predictive Models Used**
#### **XGBoost (Extreme Gradient Boosting)**
- Efficient and powerful for structured tabular data.
- Handles missing values and feature importance analysis well.
- Suitable for **short-term forecasting** of battery swap demand.

#### **LSTM (Long Short-Term Memory Networks)**
- Designed for **time-series forecasting** and capturing sequential dependencies.
- Effective for learning patterns in **battery charge cycles** and **swap trends**.
- Suitable for **long-term forecasting** and anomaly detection.

### **3. Anomaly Detection Technique: Isolation Forest**
- Detects abnormal battery behaviors by analyzing fluctuations in **State of Charge (SOC)**.
- Helps in proactive maintenance by identifying faulty batteries before failure.

---
## **How to Run the Notebook in Google Colab**

### **1. Mount Google Drive**
To access dataset files, mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### **2. Load Datasets**
Load battery and swap event data:
```python
import pandas as pd
battery_data = pd.read_csv("/content/drive/MyDrive/IoT_Battery_Analysis/data/battery_data.csv", parse_dates=['_time'])
swap_in_data = pd.read_csv("/content/drive/MyDrive/IoT_Battery_Analysis/data/swap_in_data.csv", parse_dates=['swap_in_date'])
swap_out_data = pd.read_csv("/content/drive/MyDrive/IoT_Battery_Analysis/data/swap_out_data.csv", parse_dates=['swap_out_date'])
```

### **3. Preprocess Data**
Convert timestamps and aggregate swap counts:
```python
swap_in_data['swap_in_date'] = pd.to_datetime(swap_in_data['swap_in_date'])
swap_out_data['swap_out_date'] = pd.to_datetime(swap_out_data['swap_out_date'])
swap_in_summary = swap_in_data.groupby(swap_in_data['swap_in_date'].dt.date).agg({'swap_in_count': 'sum'}).reset_index()
swap_out_summary = swap_out_data.groupby(swap_out_data['swap_out_date'].dt.date).agg({'swap_out_count': 'sum'}).reset_index()
```

### **4. Train XGBoost Model**
```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

X = battery_data[['SOC', 'Total_voltage', 'Internal_temperature_of_battery', 'Number_of_cycles']]
y = battery_data['swap_in_count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
```

### **5. Train LSTM Model**
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)
```

### **6. Evaluate Model Performance**
```python
y_pred_xgb = xgb_model.predict(X_test)
y_pred_lstm = model.predict(X_test)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)

print(f"XGBoost MAE: {mae_xgb}")
print(f"LSTM MAE: {mae_lstm}")
```

### **7. Detect Anomalies with Isolation Forest**
```python
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.05)
battery_data['anomaly'] = iso_forest.fit_predict(battery_data[['SOC', 'Total_voltage']])
```

---
## **Results & Business Impact**
- **XGBoost performed better for short-term swap demand forecasting, while LSTM excelled at long-term forecasting.**
- **Anomaly detection flagged potential battery failures, improving maintenance scheduling.**
- **Predictive insights can reduce downtime and improve operational efficiency for battery swap stations.**

---
## **Conclusion & Next Steps**
1. **Deploy predictive models in real-time monitoring systems.**
2. **Enhance feature engineering by incorporating external factors (e.g., weather, traffic).**
3. **Implement reinforcement learning for adaptive battery swap scheduling.**

This markdown provides a **structured, technical overview** of the project while detailing **how to execute the notebook** in Google Colab. 🚀

