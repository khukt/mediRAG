import streamlit as st
import json
import spacy
from spacy.cli import download

# Function to download the SpaCy model
def download_spacy_model(model_name="en_core_web_sm"):
    download(model_name)

# Check if the SpaCy model is already installed, if not, download it
model_name = "en_core_web_sm"
if not spacy.util.is_package(model_name):
    with st.spinner("Downloading SpaCy model..."):
        download_spacy_model(model_name)

# Function to load the SpaCy model with caching
@st.cache_resource
def load_spacy_model(model_name="en_core_web_sm"):
    return spacy.load(model_name)

nlp = load_spacy_model()

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

# Function to parse the query and retrieve medicine information
def parse_query(query):
    doc = nlp(query.lower())
    results = []

    for med in medicines:
        if any(token.lemma_ in med['generic_name'].lower() for token in doc) or \
           any(token.lemma_ in brand_dict[brand_id]['name'].lower() for brand_id in med['brand_names'] for token in doc) or \
           any(token.lemma_ in use.lower() for use in med['uses'] for token in doc):
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
            st.subheader(f"Generic Name: {med['generic_name']}")
            st.write('**Uses:**', ', '.join(med['uses']))
            st.write('**Side Effects:**', ', '.join(med['side_effects']))
            for brand_id in med['brand_names']:
                brand = brand_dict[brand_id]
                manufacturer = manufacturer_dict[brand['manufacturer_id']]
                st.write(f"**Brand Name:** {brand['name']}")
                st.write(f"**Dosages:** {', '.join(brand['dosages'])}")
                st.write(f"**Manufacturer:** {manufacturer['name']}")
                st.write(f"**Contact Info:** Phone: {manufacturer['contact_info']['phone']}, Email: {manufacturer['contact_info']['email']}, Address: {manufacturer['contact_info']['address']}")
    else:
        st.write('No results found.')
