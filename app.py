import streamlit as st
import pandas as pd
import joblib

# Charger le modèle
model = joblib.load("model_mathE.pkl")

# Interface utilisateur
st.title(" Prédiction de Réponses en Mathématiques")

# Charger les données d'entrée
uploaded_file = st.file_uploader(" Charger un fichier excel", type="excel")

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)

    # Vérifier la structure des données
    if "Type of Answer" in df_input.columns:
        df_input = df_input.drop(columns=["Type of Answer"])  # Supprimer la cible si elle est présente

    # Prédire
    predictions = model.predict(df_input)

    # Afficher les résultats
    df_input["Prediction"] = predictions
    st.write(" Résultats des Prédictions :")
    st.dataframe(df_input)

    # Télécharger le fichier avec les prédictions
    df_input.to_csv("predictions.csv", index=False)
    st.download_button(" Télécharger les Prédictions", "predictions.csv", "text/csv")
