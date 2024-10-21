import streamlit as st
import joblib
from PIL import Image
import numpy as np

# Charger le modèle pré-entraîné
model = joblib.load('my_model.joblib')

# Fonction pour préparer l'image (normalisation + reshape)
def preprocess_image(image):
    image = image.resize((100, 100))  # Redimensionner à (100, 100)
    image_array = np.array(image) / 255.0  # Normalisation entre 0 et 1
    reshaped_image = image_array.reshape(1, 100, 100, 3)  # Rechaper
    return reshaped_image

# Interface Streamlit
st.title("Prédiction Cat vs Dog")

uploaded_file = st.file_uploader("Importer une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Afficher l'image importée
    image = Image.open(uploaded_file)
    st.image(image, caption="Image importée", width=500)

    # Prétraiter l'image
    processed_image = preprocess_image(image)

    # Prédiction avec le modèle
    prediction = model.predict(processed_image)[0][0]  # Supposer une sortie avec une probabilité


if st.button("Prédire la Classe"):
    
    # Afficher le résultat
    if prediction > 0.5:
        st.success(f"Classe prédite : 🐱 Chat (Confiance: {prediction:.2f})")
    else:
        st.success(f"Classe prédite : 🐶 Chien (Confiance: {1 - prediction:.2f})")
