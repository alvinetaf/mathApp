import streamlit as st
import requests

st.title("Prédiction de Modèle")

# Entrée utilisateur
input_data = st.text_input("Entrez vos données séparées par des virgules:")

if st.button("Prédire"):
    # Préparer les données pour l'API
    input_list = [float(i) for i in input_data.split(',')]
    response = requests.post("http://127.0.0.1:5000/predict", json={"input": input_list})
    prediction = response.json()

    st.write(f"Prédiction: {prediction['prediction']}")
