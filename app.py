import streamlit as st
import pandas as pd
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util

# Load the transformer models
generator = pipeline('text-generation', model='gpt2', tokenizer=GPT2Tokenizer.from_pretrained('gpt2'))
retriever = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Sample expanded data with additional categories
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

def retrieve_drug_info(query, df, retriever):
    # Concatenate all relevant data into a single list for encoding
    all_texts = df['Generic Name'].tolist() + df['Commercial Names'].tolist() + df['Drug Class'].tolist() + df['Disease Names'].tolist() + df['Symptoms'].tolist() + df['Treatment Protocols'].tolist() + df['Medical Procedures'].tolist()
    
    # Encode the query and data
    query_embedding = retriever.encode(query)
    data_embeddings = retriever.encode(all_texts)

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

def generate_response(drug_info):
    if drug_info is None:
        return "No information available for this drug."

    drug_info_dict = drug_info.to_dict()
    prompt = f"Provide detailed information about the drug with the following data:\n{drug_info_dict}"

    response = generator(prompt, max_length=300, num_return_sequences=1, truncation=True, pad_token_id=50256)
    return response[0]['generated_text']

def rag_system(query, df, retriever, generator):
    drug_info = retrieve_drug_info(query, df, retriever)
    response = generate_response(drug_info)
    return response

# Streamlit app
st.title("Drug Information Retrieval and Generation")

query = st.text_input("Enter the drug name or query:")

if st.button("Get Information"):
    if query:
        response = rag_system(query, df, retriever, generator)
        st.write(response)
    else:
        st.write("Please enter a valid query.")
