import streamlit as st
import spacy
import gdown
import zipfile
import os

MODEL_ZIP = "ner_model.zip"
MODEL_DIR = "ner_model"


@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_DIR):
        # ✅ Remplace cet ID par celui de TON fichier .zip sur Google Drive
        url = "https://drive.google.com/uc?id=1MbqZc1cRHMXh_QVQ1EHk_gmva4J0jWOp"
        
        # Télécharger le modèle
        gdown.download(url, MODEL_ZIP, quiet=False)
        
        # Vérifier si c'est un vrai ZIP
        if not zipfile.is_zipfile(MODEL_ZIP):
            raise ValueError("❌ Le fichier téléchargé n'est pas un ZIP valide.")
        
        # Extraire le zip
        with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
            zip_ref.extractall(MODEL_DIR)
    
    # Charger le modèle spaCy
    return spacy.load(MODEL_DIR)

# Charger le modèle
nlp = download_and_load_model()

# Interface utilisateur Streamlit
st.title("🧠 Application de Reconnaissance d’Entités Nommées (NER)")
st.markdown("Entrez un texte pour détecter les entités nommées :")

user_input = st.text_area("✏️ Texte à analyser", height=200)

if st.button("Analyser"):
    if user_input.strip():
        doc = nlp(user_input)
        st.markdown("### 🧾 Entités reconnues :")
        for ent in doc.ents:
            st.write(f"**{ent.text}** → *{ent.label_}*")
    else:
        st.warning("Veuillez entrer un texte.")
