import streamlit as st
import gdown
import zipfile
import os
import shutil
import json
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

#https://drive.google.com/file/d/10pcCB-47bl-jQ53t8AB1_F0LD6wHNwIU/view?usp=sharing

#https://drive.google.com/file/d/1bfdQHkuHS-Y9ww9AjEslazRqEc_hL47f/view?usp=sharing

# URLs Google Drive (fichier ZIP + métriques)
ZIP_NAME = "transformer_model.zip"
EXTRACT_DIR = "transformer_model_extracted"
METRICS_FILE = "metrics_transformers.json"
DRIVE_ZIP_URL = "https://drive.google.com/uc?id=10pcCB-47bl-jQ53t8AB1_F0LD6wHNwIU"
DRIVE_METRICS_URL = "https://drive.google.com/uc?id=1bfdQHkuHS-Y9ww9AjEslazRqEc_hL47f"

# 📥 Télécharger et charger le modèle
@st.cache_resource
def load_model():
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # Télécharger le ZIP
    gdown.download(DRIVE_ZIP_URL, ZIP_NAME, quiet=False)

    # Extraire le modèle
    with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    # Charger tokenizer + modèle
    tokenizer = AutoTokenizer.from_pretrained(EXTRACT_DIR)
    model = AutoModelForTokenClassification.from_pretrained(EXTRACT_DIR)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return ner_pipeline

# 📥 Télécharger et charger les métriques
@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_FILE):
        gdown.download(DRIVE_METRICS_URL, METRICS_FILE, quiet=False)
    with open(METRICS_FILE, "r") as f:
        return json.load(f)

# Initialisation
ner = load_model()
metrics = load_metrics()

# Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Aller à :", ["🏠 Accueil", "📊 Performances", "📝 Détection d'entités"])

# Pages
if page == "🏠 Accueil":
    st.title("🔍 Détection d'entités nommées (NER)")
    st.markdown("""
    Ce modèle de Reconnaissance d'Entités Nommées (NER) utilise l'architecture Transformers pour identifier les entités dans vos textes.

    **Technologie utilisée :** Hugging Face Transformers  
    **Source :** Modèle entraîné sur des données personnalisées
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
            results = ner(text)
            st.markdown("### 📌 Entités détectées :")
            for ent in results:
                st.write(f"**{ent['word']}** → *{ent['entity_group']}* ({ent['score']:.2f})")
        else:
            st.warning("Veuillez entrer un texte.")
