import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import joblib

# Custom CSS for gradient background
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to bottom, #ffe6f0 0%, #ffffff 100%);
    }
    .main {
        background: linear-gradient(to bottom, #ffe6f0 0%, #ffffff 100%);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load data
@st.cache_data
def load_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    return train_df, test_df

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('demand_forecast_model.pkl')
    return model

# Forecast function
def forecast_demand(model, features):
    prediction = model.predict(features)
    return prediction

# App Layout
st.image("https://images.unsplash.com/photo-1594007654729-232d177cb92c?auto=format&fit=crop&w=1470&q=80", use_column_width=True)

st.title("ForecastX by Chidima 🚴")
st.markdown("## *Predicting Tomorrow's Rides with Style and Precision.*")

train_df, test_df = load_data()
model = load_model()

# Sidebar Inputs
st.sidebar.title("Adjust Input Parameters ⚙️")
hour = st.sidebar.slider("Select the Hour of Day", 0, 23, 12)
day_of_week = st.sidebar.selectbox("Select the Day of Week", train_df['day_of_week'].unique())
weather_condition = st.sidebar.selectbox("Select Weather Condition", ['Clear', 'Cloudy', 'Rainy', 'Stormy'])

# Prepare Features for Prediction
input_df = pd.DataFrame({
    'hour': [hour],
    'day_of_week': [day_of_week],
    'weather_condition': [weather_condition]
})

# Encoding if necessary (assuming weather_condition needs encoding)
input_df = pd.get_dummies(input_df)

# Match model expected columns
model_features = model.feature_names_in_
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_features]

if st.sidebar.button("Predict Demand"):
    demand_prediction = forecast_demand(model, input_df)
    st.success(f"Predicted Bike Ride Demand: {demand_prediction[0]:.2f}")

# Demand Trends
st.write("**Demand Trends**")
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(train_df['hour'], train_df['demand'], marker='o', linestyle='-', color='deeppink')
ax.set_xlabel("Hour of the Day")
ax.set_ylabel("Demand")
ax.set_title("Bike Ride Demand Over Time")
ax.grid(True)
st.pyplot(fig)

# Model Evaluation
st.write("**Model Evaluation**")
X_test = test_df.drop(['demand'], axis=1)
y_test = test_df['demand']
model_score = model.score(X_test, y_test)
st.write(f"Model R² Score: {model_score:.2f}")

# Contact Section
st.sidebar.title("Contact 📞")
st.sidebar.write("For inquiries, please contact: support@forecastx.com")
st.sidebar.write("Follow us on [LinkedIn](https://www.linkedin.com)")
st.sidebar.write("Follow us on [Twitter](https://twitter.com)")

