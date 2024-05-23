import streamlit as st
import json
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Function to load JSON data with caching
@st.cache_data
def load_json_data(file_path):
    with open(file_path) as f:
        return json.load(f)

medicines = load_json_data('medicines.json')['medicines']
brand_names = load_json_data('brand_names.json')['brand_names']
manufacturers = load_json_data('manufacturers.json')['manufacturers']

# Create dictionaries for quick lookups
brand_dict = {brand['id']: brand for brand in brand_names}
manufacturer_dict = {manufacturer['id']: manufacturer for manufacturer in manufacturers}

# Load the multilingual NLP model with caching
@st.cache_resource
def load_nlp_model():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
    return tokenizer, model

tokenizer, model = load_nlp_model()

# Function to parse the query and retrieve medicine information
def parse_query(query):
    # Tokenize and encode the query using the model
    inputs = tokenizer(query, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    
    results = []

    for med in medicines:
        generic_match = any(token in med['generic_name'].lower() for token in tokens) or \
                        any(token in med.get('generic_name_mm', '').lower() for token in tokens)
        brand_match = any(any(token in brand_dict[brand_id]['name'].lower() for token in tokens) for brand_id in med['brand_names'])
        uses_match = any(any(token in use.lower() for token in tokens) for use in med['uses']) or \
                     any(any(token in use.lower() for token in tokens) for use in med.get('uses_mm', []))
        
        if generic_match or brand_match or uses_match:
            results.append(med)
    
    return results

# Streamlit app
st.title('Medicine Information Retrieval')

st.write('Enter your query about a medicine name (generic or brand) or a symptom.')

query = st.text_input('Query')

if query:
    results = parse_query(query)
    if results:
        for med in results:
            st.subheader(f"Generic Name: {med['generic_name']} ({med.get('generic_name_mm', '')})")
            st.write('**Uses:**', ', '.join(med['uses']) + ', ' + ', '.join(med.get('uses_mm', [])))
            st.write('**Side Effects:**', ', '.join(med['side_effects']) + ', ' + ', '.join(med.get('side_effects_mm', [])))
            for brand_id in med['brand_names']:
                brand = brand_dict[brand_id]
                manufacturer = manufacturer_dict[brand['manufacturer_id']]
                st.write(f"**Brand Name:** {brand['name']}")
                st.write(f"**Dosages:** {', '.join(brand['dosages'])}")
                st.write(f"**Manufacturer:** {manufacturer['name']}")
                st.write(f"**Contact Info:** Phone: {manufacturer['contact_info']['phone']}, Email: {manufacturer['contact_info']['email']}, Address: {manufacturer['contact_info']['address']}")
    else:
        st.write('No results found.')
