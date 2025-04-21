import streamlit as st
import spacy
import gdown
import zipfile
from pathlib import Path

# Chemin du modèle dans Google Drive (utilisation de gdown)
GDRIVE_URL = "https://drive.google.com/drive/folders/1CXsQCyrKzGzcoF-p5RptpVh7Ix8q8V4W?usp=sharing"
MODEL_DIR = Path("model-best")

@st.cache_resource
def download_and_load_model():
    zip_path = "ner_model.zip"
    
    # Télécharger le modèle
    if not Path(zip_path).exists():
        st.info("Téléchargement du modèle...")
        gdown.download(GDRIVE_URL, zip_path, quiet=False)

    # Extraire l'archive
    if not MODEL_DIR.exists():
        st.info("Extraction du modèle...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(MODEL_DIR)

    # Charger le modèle spaCy
    st.success("Modèle chargé avec succès.")
    return spacy.load(MODEL_DIR)

# UI
st.title("🧠 Application NER avec spaCy")
st.write("Entrez un texte pour détecter les entités nommées :")

text_input = st.text_area("Texte à analyser", "Barack Obama was born in Hawaii and was elected president in 2008.")

if st.button("Analyser"):
    nlp = download_and_load_model()
    doc = nlp(text_input)

    st.subheader("🟢 Entités détectées :")
    for ent in doc.ents:
        st.markdown(f"- **{ent.text}** ({ent.label_})")

    st.subheader("🔍 Texte avec entités surlignées :")
    st.markdown(spacy.displacy.render(doc, style="ent", page=True), unsafe_allow_html=True)
