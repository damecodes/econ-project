import streamlit as st
from fred_data import get_data
from model import train_model, predict

st.title("GDP Nowcasting Dashboard")

df = get_data()
df["GDP_Growth"] = df["GDP"].pct_change() * 100
df = df.dropna()

model = train_model(df)
df = predict(model, df)

st.line_chart(df[["GDP_Growth", "Predicted_GDP_Growth"]])

st.write("### Model Summary")
st.text(model.summary())