import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("/Users/gnwuha/Documents/AI_ML/projects/FocusScore/focus_data.csv")

X = data.drop('focus', axis=1)
y = data['focus']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,random_state=42)
    
model = DecisionTreeClassifier(max_depth=15)

model.fit(X_train, y_train)
predictions = model.predict(X_val)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

import streamlit as st
import pandas as pd


st.title("🔍 FocusScore Predictor")

st.markdown("Enter your lifestyle information below to get a prediction on your focus level.")

# Example features (update based on your dataset!)
sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
screen_time = st.slider("Screen Time (hours/day)", 0.0, 16.0, 6.0)
water_intake = st.slider("Water Intake (liters/day)", 0.0, 5.0, 2.0)

# Create input DataFrame
input_data = pd.DataFrame({
    'sleep_hours': [sleep_hours],
    'stress_level': [stress_level],
    'screen_time': [screen_time],
    'water_intake': [water_intake]
})

if st.button("Predict Focus"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Focus Level: **{prediction}**")

    # Simple feedback
    if prediction == 0:
        st.warning("⚠️ You may be at risk of low focus. Try getting more sleep, reducing screen time, or managing stress.")
    else:
        st.info("✅ You're predicted to have good focus. Keep maintaining healthy habits!")