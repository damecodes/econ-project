import streamlit as st
from fred_data import get_data
from data_processing import compute_gdp_growth, compute_feature_growth
from model import train_model, predict

st.title("GDP Nowcasting Dashboard")

df = get_data()

df = compute_gdp_growth(df)
df = compute_feature_growth(df)


model, scaler = train_model(df)
df = predict(model, scaler, df)

# Show chart
st.line_chart(df[["GDP_Growth", "Predicted_GDP_Growth"]])

# Debug: check if dates extend into 2026
st.write("### Latest Data")
st.write(df.tail(10))

# Model summary
st.write("### Model Summary")
st.text(model.summary())

st.write("NaNs in dataset:")
st.write(df.isna().sum())