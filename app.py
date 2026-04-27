import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from fred_data import get_data
from data_processing import compute_gdp_growth, compute_feature_growth
from model import train_model, predict

st.title("GDP Nowcasting Dashboard")

# 1. Load data FIRST
df = get_data()

# 2. Process data
df = compute_gdp_growth(df)
df = compute_feature_growth(df)

# 3. Train model
model, scaler = train_model(df)

# 4. Predict full dataset
df = predict(model, scaler, df)

# 5. NOW you can use df safely
latest_features = df[["Unemployment", "CPI", "Industrial_Production", "Retail_Sales"]].iloc[-1:]

latest_features = latest_features.replace([np.inf, -np.inf], np.nan)
latest_features = latest_features.ffill().bfill()

latest_scaled = scaler.transform(latest_features)
latest_scaled = pd.DataFrame(latest_scaled, columns=latest_features.columns)

latest_scaled = sm.add_constant(latest_scaled, has_constant='add')
latest_scaled = latest_scaled[model.model.exog_names]

latest_prediction = model.predict(latest_scaled)[0]

# 6. Plot
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(df.index, df["GDP_Growth"], label="Actual GDP Growth")
ax.plot(df.index, df["Predicted_GDP_Growth"], linestyle="dashed", label="Predicted GDP Growth")

ax.set_title("GDP Nowcasting Model")
ax.legend()

st.pyplot(fig)

# 7. Model summary
st.subheader("Model Summary")
st.text(model.summary())

# 8. Nowcast metric
st.subheader("Current GDP Nowcast")
st.metric(
    label="Predicted GDP Growth (Annualized %)",
    value=f"{latest_prediction:.2f}%"
)