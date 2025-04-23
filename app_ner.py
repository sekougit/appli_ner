import streamlit as st
import spacy
import gdown
import zipfile
import os
import shutil
import json

#https://drive.google.com/file/d/1MCEW0B8N7I-AJl7zw7h23zRwgnsaD-ai/view?usp=sharing

#https://drive.google.com/file/d/1IJFnBrHrEOL4oaIaaa7ENgbahTICJuH7/view?usp=sharing

ZIP_NAME = "ner_model.zip"
EXTRACT_DIR = "ner_model_extracted"
METRICS_FILE = "metrics.json"
DRIVE_ZIP_URL = "https://drive.google.com/uc?id=1MCEW0B8N7I-AJl7zw7h23zRwgnsaD-ai"
DRIVE_METRICS_URL = "https://drive.google.com/uc?id=1IJFnBrHrEOL4oaIaaa7ENgbahTICJuH7"  # Remplace par le bon ID

# 📥 Télécharger et charger le modèle
@st.cache_resource
def load_model():
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # Télécharger le modèle ZIP
    gdown.download(DRIVE_ZIP_URL, ZIP_NAME, quiet=False)

    # Extraire
    with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    # Chercher le dossier avec meta.json
    for root, dirs, files in os.walk(EXTRACT_DIR):
        if "meta.json" in files:
            return spacy.load(root)

    raise FileNotFoundError("❌ Fichier meta.json non trouvé.")

# 📥 Télécharger et charger les métriques
@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_FILE):
        gdown.download(DRIVE_METRICS_URL, METRICS_FILE, quiet=False)
    with open(METRICS_FILE, "r") as f:
        return json.load(f)

# Initialisation
nlp = load_model()
metrics = load_metrics()

# Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Aller à :", ["🏠 Accueil", "📊 Performances", "📝 Détection d'entités"])

# Pages
if page == "🏠 Accueil":
    st.title("🔍 Détection d'entités nommées (NER)")
    st.markdown("""
    Ce modèle de Reconnaissance d'Entités Nommées (NER) a été entraîné pour détecter des entités spécifiques dans vos textes.
    
    **Sources :** données personnelles – entraîné avec spaCy.
    """)

elif page == "📊 Performances":
    st.title("📈 Performances du modèle")

    precision = metrics['ents_p']
    recall = metrics['ents_r']
    f1 = metrics['ents_f']
    accuracy = (precision + recall) / 2

    st.metric("🎯 Précision", f"{precision:.2f}%")
    st.metric("📥 Rappel", f"{recall:.2f}%")
    st.metric("⚖️ F1-Score", f"{f1:.2f}%")
    st.metric("⭐ Accuracy", f"{accuracy:.2f}%")


elif page == "📝 Détection d'entités":
    st.title("📝 Entrez un texte à analyser")
    text = st.text_area("Tapez ici votre texte...")
    if st.button("Analyser"):
        if text.strip():
            doc = nlp(text)
            st.markdown("### 📌 Entités détectées :")
            for ent in doc.ents:
                st.write(f"**{ent.text}** → *{ent.label_}*")
        else:
            st.warning("Veuillez entrer un texte.")
