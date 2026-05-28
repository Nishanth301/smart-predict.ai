import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# Page settings
st.set_page_config(
    page_title="SmartPredict AI",
    layout="wide"
)

# Load dataset
data = pd.read_csv("data.csv")

# Train AI
X = data[["temperature", "vibration"]]
y = data["status"]

model = RandomForestClassifier()
model.fit(X, y)

# Title
st.title("SmartPredict AI")
st.subheader("AI-Powered Machine Health Monitoring Dashboard")

# Create columns
col1, col2 = st.columns(2)

# Temperature input
with col1:
    temp = st.slider(
        "Temperature (°C)",
        20,
        100,
        40
    )

# Vibration input
with col2:
    vibration = st.slider(
        "Vibration Level",
        0.0,
        3.0,
        0.5
    )

# Prediction
prediction = model.predict([[temp, vibration]])
probabilities = model.predict_proba([[temp, vibration]])

status = prediction[0]
confidence = max(probabilities[0]) * 100

# Status box
st.divider()

if status == "healthy":
    st.success(f"Machine Status: HEALTHY")

elif status == "warning":
    st.warning(f"Machine Status: WARNING")

else:
    st.error(f"Machine Status: DANGER")

# Metrics
metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "Temperature",
    f"{temp} °C"
)

metric2.metric(
    "Vibration",
    f"{vibration}"
)

metric3.metric(
    "AI Confidence",
    f"{confidence:.2f}%"
)

# Divider
st.divider()

# Graph title
st.subheader("Machine Health Dataset")

# Interactive graph
fig = px.scatter(
    data,
    x="temperature",
    y="vibration",
    color="status",
    title="Temperature vs Vibration",
    hover_data=["status"]
)

st.plotly_chart(fig, use_container_width=True)

# Footer
st.divider()

st.caption(
    "Built using Python, Streamlit, Machine Learning, and Predictive Maintenance concepts."
)