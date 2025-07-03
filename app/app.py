import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load model and scaler
base_path = "app/" if os.path.exists("app/xgboost_model.pkl") else ""

model = joblib.load(os.path.join(base_path, "xgboost_model.pkl"))
scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
red = pd.read_csv(os.path.join(base_path, "winequality-red.csv"), sep=';')
white = pd.read_csv(os.path.join(base_path, "winequality-white.csv"), sep=';')
full_data = pd.concat([red, white], ignore_index=True)

st.set_page_config(page_title="🍷 Wine Quality Predictor", layout="centered")

st.title("🍷 Wine Quality Prediction App")
st.markdown("Predict whether a wine is **Bad**, **Average**, or **Good** based on its properties using a trained XGBoost model.")

st.sidebar.header("Input Wine Features")

# Use feature columns except target
feature_names = full_data.columns[:-1]

def get_slider(name):
    min_val = float(full_data[name].min())
    max_val = float(full_data[name].max())
    mean_val = float(full_data[name].mean())
    return st.sidebar.slider(
        label=name,
        min_value=round(min_val, 2),
        max_value=round(max_val, 2),
        value=round(mean_val, 2)
    )

# Take user input
user_inputs = [get_slider(name) for name in feature_names]
input_df = pd.DataFrame([user_inputs], columns=feature_names)

# Scale the input
scaled_input = scaler.transform(input_df)

# Make prediction
pred_class = model.predict(scaled_input)[0]
pred_proba = model.predict_proba(scaled_input)[0]

label_map = {
    0: "Bad Quality (3–5)",
    1: "Average Quality (6)",
    2: "Good Quality (7–8)"
}

# Show prediction result
st.subheader("🔍 Prediction Result")
st.success(f"**Predicted Quality:** {label_map[pred_class]}")

# Show probability bar chart
st.subheader("📈 Prediction Confidence")
conf_df = pd.DataFrame({
    "Class": [label_map[i] for i in range(3)],
    "Probability": pred_proba
})
conf_df_sorted = conf_df.sort_values(by="Probability", ascending=True)

fig, ax = plt.subplots(figsize=(6, 3))
sns.barplot(data=conf_df_sorted, x="Probability", y="Class", palette="coolwarm", ax=ax)
ax.set_xlim(0, 1)
st.pyplot(fig)








