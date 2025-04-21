import streamlit as st
import spacy

# Charger le modèle
@st.cache_resource
def load_model():
    return spacy.load("ner_output")

nlp = load_model()

st.title("Reconnaissance d'Entités Nommées (NER)")
user_input = st.text_area("Entrez un texte ici pour la détection NER :")

if st.button("Analyser"):
    if user_input:
        doc = nlp(user_input)
        st.subheader("Entités trouvées :")
        for ent in doc.ents:
            st.markdown(f"- **{ent.text}** : {ent.label_}")
    else:
        st.warning("Veuillez entrer un texte.")
