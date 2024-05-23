import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator
from functools import lru_cache

# Initialize the translator
translator = Translator()

# Load the transformer model with caching to improve performance
@lru_cache(maxsize=1)
def load_model():
    retriever = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return retriever

retriever = load_model()

# Sample data with additional categories
data = {
    "Generic Name": ["Ibuprofen", "Paracetamol"],
    "Commercial Names": ["Advil, Motrin", "Tylenol"],
    "Usage": ["Pain, Inflammation", "Pain, Fever"],
    "Dosage": ["200-400 mg every 4-6 hours", "500-1000 mg every 4-6 hours"],
    "Side Effects": ["Nausea, Dizziness", "Nausea, Rash"],
    "Contraindications": ["Peptic Ulcer, Renal Impairment", "Liver Disease"],
    "Interactions": ["Anticoagulants", "Alcohol"],
    "Mechanism of Action": ["Inhibits COX-1 and COX-2", "Inhibits prostaglandin synthesis"],
    "Pharmacokinetics": ["Oral, hepatic metabolism", "Oral, hepatic metabolism"],
    "Warnings and Precautions": ["Gastrointestinal bleeding risk", "Hepatotoxicity risk"],
    "Drug Class": ["NSAID", "Analgesic"],
    "Disease Names": ["Osteoarthritis, Rheumatoid arthritis", "Fever, Headache"],
    "Symptoms": ["Pain, Swelling", "Fever, Pain"],
    "Treatment Protocols": ["First-line treatment for pain", "First-line treatment for fever"],
    "Medical Procedures": ["None", "None"]
}

df = pd.DataFrame(data)

# Precompute embeddings for all texts in the dataframe
@st.cache_data
def precompute_embeddings():
    all_texts = df['Generic Name'].tolist() + df['Commercial Names'].tolist() + df['Drug Class'].tolist() + df['Disease Names'].tolist() + df['Symptoms'].tolist() + df['Treatment Protocols'].tolist() + df['Medical Procedures'].tolist()
    data_embeddings = retriever.encode(all_texts, convert_to_tensor=True)
    return all_texts, data_embeddings

all_texts, data_embeddings = precompute_embeddings()

def retrieve_drug_info(query, all_texts, data_embeddings):
    # Encode the query
    query_embedding = retriever.encode(query, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(query_embedding, data_embeddings)
    best_match_idx = similarities.argmax().item()

    # Retrieve the best matching row
    num_records = len(df)
    if best_match_idx < num_records:
        return df.iloc[best_match_idx]
    elif best_match_idx < 2 * num_records:
        return df.iloc[best_match_idx - num_records]
    elif best_match_idx < 3 * num_records:
        return df.iloc[best_match_idx - 2 * num_records]
    elif best_match_idx < 4 * num_records:
        return df.iloc[best_match_idx - 3 * num_records]
    elif best_match_idx < 5 * num_records:
        return df.iloc[best_match_idx - 4 * num_records]
    else:
        return df.iloc[best_match_idx - 5 * num_records]

def get_suggestions(prefix, terms):
    return [term for term in terms if term.lower().startswith(prefix.lower())]

# Streamlit app
st.title("Drug Information Retrieval")

# Language selection
language = st.selectbox("Select language:", ["English", "Myanmar (Burmese)"])

# Input with suggestions
query_prefix = st.text_input("Enter the drug name or query:")

suggestions = []
if query_prefix:
    suggestions = get_suggestions(query_prefix, all_texts)

if suggestions:
    query = st.selectbox("Did you mean:", suggestions)
else:
    query = query_prefix

if st.button("Get Information"):
    if query:
        try:
            if language == "Myanmar (Burmese)":
                try:
                    query = translator.translate(query, src='my', dest='en').text
                except Exception as e:
                    st.write(f"Translation error: {e}")
                    query = query_prefix  # fallback to original query if translation fails
                
                drug_info = retrieve_drug_info(query, all_texts, data_embeddings)
                
                try:
                    response = translator.translate(str(drug_info.to_dict()), src='en', dest='my').text
                except Exception as e:
                    st.write(f"Translation error: {e}")
                    response = str(drug_info.to_dict())  # fallback to English response if translation fails
            else:
                drug_info = retrieve_drug_info(query, all_texts, data_embeddings)
                response = str(drug_info.to_dict())
            st.write(response)
        except Exception as e:
            st.write(f"An error occurred: {e}")
    else:
        st.write("Please enter a valid query.")
