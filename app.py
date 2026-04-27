import streamlit as st
import matplotlib.pyplot as plt

from fred_data import get_data
from data_processing import compute_gdp_growth, compute_feature_growth
from model import train_model, predict

st.title("GDP Nowcasting Dashboard")

# Load and process data
df = get_data()
df = compute_gdp_growth(df)
df = compute_feature_growth(df)

# Train model and predict
model, scaler = train_model(df)
df = predict(model, scaler, df)

# Plot results
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(df.index, df["GDP_Growth"], label="Actual GDP Growth")
ax.plot(
    df.index,
    df["Predicted_GDP_Growth"],
    linestyle="dashed",
    label="Predicted GDP Growth"
)

ax.set_title("GDP Nowcasting Model (Extended Forecast)")
ax.legend()

st.pyplot(fig)

# Model summary
st.subheader("Model Summary")
st.text(model.summary())