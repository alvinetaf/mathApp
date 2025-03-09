
import streamlit as st
import pandas as pd
import joblib

# 📌 Charger le modèle et le scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🔮 Prédiction de Réponses MathE")

# 📂 Upload du fichier utilisateur
uploaded_file = st.file_uploader("📥 Charger un fichier CSV", type="csv")

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)

    # Prétraitement des données
    if "Type of Answer" in df_input.columns:
        df_input = df_input.drop(columns=["Type of Answer"])
    df_input_scaled = scaler.transform(df_input)

    # 🔮 Prédictions
    predictions = model.predict(df_input_scaled)

    # 📊 Afficher les résultats
    df_input["Prediction"] = predictions
    st.dataframe(df_input)

    # 📥 Télécharger les résultats
    df_input.to_csv("predictions.csv", index=False)
    st.download_button("📥 Télécharger les Prédictions", open("predictions.csv", "rb"), "predictions.csv", "text/csv")
