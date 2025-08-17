import pandas as pd
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
import pickle
from tensorflow.keras.models import load_model

class RandomAppModel:
    def __init__(self, app_list):
        self.app_list = app_list

    def predict(self, user_id=None):
        import random
        return random.choice(self.app_list)



import pickle

with open("model/next_app_pred_model.pkl", "rb") as f:
    next_app_model = pickle.load(f)



# ─────────────────────────────────────────────
# 🧠 Battery Drain Transformer Model
# ─────────────────────────────────────────────
class TransformerRegressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=2)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, seq=1, features)
        x = self.transformer(x)
        return self.fc(x.squeeze(1))

# ─────────────────────────────────────────────
# 🧠 Addiction Risk Transformer Model
# ─────────────────────────────────────────────
class AddictionTransformer(nn.Module):
    def __init__(self, input_dim=4, embed_dim=32, nhead=2, num_layers=2, num_classes=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

# ─────────────────────────────────────────────
# 📥 Load Models and Scalers
# ─────────────────────────────────────────────
scaler_X = joblib.load("model/x_scaler.pkl")
scaler_y = joblib.load("model/y_scaler.pkl")
device_model_encoder = joblib.load("model/Device Model_encoder.pkl")
os_encoder = joblib.load("model/Operating System_encoder.pkl")
gender_encoder = joblib.load("model/Gender_encoder.pkl")

battery_model = TransformerRegressor(input_size=8)
battery_model.load_state_dict(torch.load("model/transformer_model.pt", map_location="cpu"))
battery_model.eval()

addiction_model = AddictionTransformer()
addiction_model.load_state_dict(torch.load("model/addiction_transformer.pt", map_location="cpu"))
addiction_model.eval()

# Load behavior model
behavior_model = load_model("user_behavior_transformer_model.h5", compile=False)
enc_dev = joblib.load("enc_device.joblib")
enc_os = joblib.load("enc_os.joblib")
enc_gender = joblib.load("enc_gender.joblib")
scaler_static = joblib.load("scaler_static.joblib")
uid_encoder = joblib.load("enc_user_id.joblib")

class_labels = {
    0: "Light User", 1: "Moderate User", 2: "Social Media Centric",
    3: "Gamer", 4: "Productivity-Focused", 5: "Unclassified"
}

# Load Next App Prediction Model
with open("model/next_app_pred_model.pkl", "rb") as f:
    next_app_model = pickle.load(f)

# ─────────────────────────────────────────────
# 📊 Load Dataset
# ─────────────────────────────────────────────
df_main = pd.read_csv("expanded_user_behavior_dataset.csv")
df_main.rename(columns={'User ID': 'user_id'}, inplace=True)

seq_data = np.load("model/addiction_sequences.npz")
X_seq_all = seq_data['X']

screen_data = np.load("model/screen_time_labels.npz")
avg_screen_time_all = screen_data["avg_screen_time"]
true_labels = screen_data["labels"]

# ─────────────────────────────────────────────
# 🎛️ Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="🔋 Battery + Addiction Predictor", page_icon="📱")
st.title("📱 Mobile Usage & 🔋 Battery Drain Predictor")

user_input = st.text_input("🔢 Enter User ID (numeric)", "101")

if user_input.isdigit():
    user_id = int(user_input)

    if user_id in df_main['user_id'].values:
        idx = df_main[df_main['user_id'] == user_id].index[0]
        user_row = df_main.iloc[idx]
        seq = X_seq_all[idx]
        avg_screen_time = avg_screen_time_all[idx]

        st.subheader("👤 Auto-Filled User Info")
        with st.form("predict_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("📱 Device Model", user_row['Device Model'], disabled=True)
                st.text_input("🕒 Screen On Time", str(user_row['Screen On Time (hours/day)']), disabled=True)
                st.text_input("📲 Apps Installed", str(user_row['Number of Apps Installed']), disabled=True)
                st.text_input("🌐 Data Usage", str(user_row['Data Usage (MB/day)']), disabled=True)
            with col2:
                st.text_input("💻 OS", user_row['Operating System'], disabled=True)
                st.text_input("📱 App Usage", str(user_row['App Usage Time (min/day)']), disabled=True)
                st.text_input("🎂 Age", str(user_row['Age']), disabled=True)
                st.text_input("👤 Gender", user_row['Gender'], disabled=True)

            submitted = st.form_submit_button("🚀 Predict")

        if submitted:
            # 📱 Predict Behavior Class
            dev_enc = enc_dev.transform([user_row['Device Model']])[0]
            os_enc = enc_os.transform([user_row['Operating System']])[0]
            gender_enc = enc_gender.transform([user_row['Gender']])[0]

            static_input = np.array([[dev_enc, os_enc, gender_enc,
                                      user_row['App Usage Time (min/day)'],
                                      user_row['Screen On Time (hours/day)'],
                                      user_row['Battery Drain (mAh/day)'],
                                      user_row['Number of Apps Installed'],
                                      user_row['Data Usage (MB/day)'],
                                      user_row['Age']]])
            static_scaled = scaler_static.transform(static_input)
            uid_enc = uid_encoder.transform([user_id])
            dummy_seq = np.random.rand(1, 7, 3)

            probs = behavior_model.predict([uid_enc, static_scaled, dummy_seq])
            behavior_class = int(np.argmax(probs))
            st.subheader("📊 Behavior Classification")
            st.success(f"Predicted Class: **{class_labels[behavior_class]}** (Class {behavior_class})")

            # 🔋 Battery Prediction
            dev = device_model_encoder.transform([user_row['Device Model']])[0]
            os_ = os_encoder.transform([user_row['Operating System']])[0]
            gender = gender_encoder.transform([user_row['Gender']])[0]

            X_input = np.array([[user_row['App Usage Time (min/day)'],
                                 user_row['Screen On Time (hours/day)'],
                                 user_row['Number of Apps Installed'],
                                 user_row['Data Usage (MB/day)'],
                                 user_row['Age'], dev, os_, gender]])
            X_scaled = scaler_X.transform(X_input)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            with torch.no_grad():
                output = battery_model(X_tensor).numpy()
                prediction = scaler_y.inverse_transform(output)

            # 📵 Addiction Risk
            seq_tensor = torch.tensor(seq.reshape(1, 7, 4), dtype=torch.float32)
            with torch.no_grad():
                logits = addiction_model(seq_tensor)
                pred_class = torch.argmax(logits, dim=1).item()

            risk_labels = ["✅ Low", "⚠️ Moderate", "🚨 High"]

            st.subheader("📱 Average Screen-On Time (7 Days)")
            st.metric("Avg Screen Time", f"{avg_screen_time:.2f} hrs/day")

            st.subheader("📵 Screen Addiction Risk")
            st.info(f"Predicted Risk Level: {risk_labels[pred_class]}")

            st.success(f"🔋 Estimated Battery Drain: **{prediction[0][0]:.2f} mAh/day**")

            # 🎯 Next App Prediction
            next_app = next_app_model.predict(user_id)
            st.subheader("📲 Next App Prediction")
            st.success(f"Next App the User is Likely to Open: **{next_app}**")

    else:
        st.warning("⚠️ User ID not found in dataset.")
else:
    st.info("ℹ️ Please enter a valid numeric User ID.")


# Read and display accuracy from text file
with open("model_accuracy.txt", "r") as f:
    accuracy_text = f.read()

st.subheader("📊 Model Accuracy Metrics")
st.text(accuracy_text)

