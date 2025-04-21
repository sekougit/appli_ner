import streamlit as st
import spacy
import gdown
import os

# Titre de l'application
st.title("🔍 Application NER avec spaCy")
#1CXsQCyrKzGzcoF-p5RptpVh7Ix8q8V4W

# ID du fichier .spacy sur Google Drive
file_id = "1CXsQCyrKzGzcoF-p5RptpVh7Ix8q8V4W"  # <- Remplace par ton propre ID
output_path = "ner_model/model-best.spacy"

# Télécharger le modèle depuis Google Drive si non existant
if not os.path.exists(output_path):
    st.info("📥 Téléchargement du modèle depuis Google Drive...")
    url = f"https://drive.google.com/drive/folders/1CXsQCyrKzGzcoF-p5RptpVh7Ix8q8V4W?usp=sharing"
    os.makedirs("ner_model", exist_ok=True)
    gdown.download(url, output_path, quiet=False)

# Charger le modèle
@st.cache_resource
def load_model():
    return spacy.load(output_path)

nlp = load_model()

# Zone de texte
text = st.text_area("Entrez un texte à analyser :", "Barack Obama was born in Hawaii.")

# Analyse du texte
if st.button("Analyser"):
    doc = nlp(text)
    st.subheader("📄 Résultats de la Reconnaissance d'Entités Nommées")
    
    if doc.ents:
        for ent in doc.ents:
            st.markdown(f"**{ent.text}** → `{ent.label_}`")
    else:
        st.write("Aucune entité reconnue.")
