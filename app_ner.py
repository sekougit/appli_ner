import streamlit as st
import spacy
import gdown
import zipfile
import os
import shutil

ZIP_NAME = "ner_model.zip"
EXTRACT_DIR = "ner_model_extracted"

@st.cache_resource
def download_and_load_model():
    # Supprimer les anciens fichiers
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # ✅ Remplacer ici par ton vrai ID de modèle Google Drive
    gdown.download("https://drive.google.com/uc?id=1MbqZc1cRHMXh_QVQ1EHk_gmva4J0jWOp", ZIP_NAME, quiet=False)

    # Vérifie que c’est bien un zip valide
    if not zipfile.is_zipfile(ZIP_NAME):
        raise ValueError("❌ Le fichier téléchargé n'est pas un ZIP valide.")

    # Extraire le zip
    with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    # Chercher récursivement le dossier contenant meta.json
    for root, dirs, files in os.walk(EXTRACT_DIR):
        if "meta.json" in files:
            st.success(f"✅ Modèle trouvé dans : {root}")
            return spacy.load(root)

    raise FileNotFoundError("❌ Aucun fichier meta.json trouvé dans le ZIP.")

# Charger le modèle
nlp = download_and_load_model()

# Interface utilisateur
st.title("🔍 Détection d'entités nommées (NER)")
text = st.text_area("✍️ Entrez un texte pour détecter les entités :")

if st.button("Analyser"):
    if text.strip():
        doc = nlp(text)
        st.markdown("### 📌 Entités détectées :")
        for ent in doc.ents:
            st.write(f"**{ent.text}** → *{ent.label_}*")
    else:
        st.warning("Veuillez entrer un texte.")
