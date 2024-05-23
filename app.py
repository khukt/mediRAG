import streamlit as st
import json
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Set page configuration
st.set_page_config(page_title="Medicine Information Retrieval", layout="wide")

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
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    model = AutoModelForMaskedLM.from_pretrained("distilbert-base-multilingual-cased")
    return tokenizer, model

tokenizer, model = load_nlp_model()

# Function to parse the query and retrieve medicine information
def parse_query(query):
    query = query.lower()
    results = []

    for med in medicines:
        if (query in med['generic_name'].lower() or
            query in med.get('generic_name_mm', '').lower() or
            any(query in use.lower() for use in med['uses']) or
            any(query in use.lower() for use in med.get('uses_mm', [])) or
            any(query in effect.lower() for effect in med['side_effects']) or
            any(query in effect.lower() for effect in med.get('side_effects_mm', [])) or
            any(query in brand_dict[brand_id]['name'].lower() for brand_id in med['brand_names'])):
            results.append(med)
    
    return results

# Sidebar with instructions
st.sidebar.title("Instructions")
st.sidebar.write("""
Enter your query about a medicine name (generic or brand) or a symptom in the input box. 
The app will display relevant medicine information based on your query. You can search in either English or Burmese.
""")

st.title('Medicine Information Retrieval')

st.write('Enter your query about a medicine name (generic or brand) or a symptom.')

query = st.text_input('Query')

if query:
    results = parse_query(query)
    if results:
        for med in results:
            st.markdown(f"### {med['generic_name']} ({med.get('generic_name_mm', '')})")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Uses**")
                st.write(', '.join(med['uses']))
                if 'uses_mm' in med:
                    st.write(', '.join(med['uses_mm']))
            
            with col2:
                st.markdown("**Side Effects**")
                st.write(', '.join(med['side_effects']))
                if 'side_effects_mm' in med:
                    st.write(', '.join(med['side_effects_mm']))
            
            st.markdown("**Brands and Dosages**")
            for brand_id in med['brand_names']:
                brand = brand_dict[brand_id]
                manufacturer = manufacturer_dict[brand['manufacturer_id']]
                st.markdown(f"**{brand['name']}**")
                st.write(f"**Dosages:** {', '.join(brand['dosages'])}")
                st.write(f"**Manufacturer:** {manufacturer['name']}")
                st.write(f"**Contact Info:** Phone: {manufacturer['contact_info']['phone']}, Email: {manufacturer['contact_info']['email']}, Address: {manufacturer['contact_info']['address']}")
                st.markdown("---")
    else:
        st.write('No results found.')
