import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
import psutil
import torch
import pandas as pd

# Function to print memory usage in GB
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    st.write(f"RSS: {mem_info.rss / (1024 ** 3):.2f} GB, VMS: {mem_info.vms / (1024 ** 3):.2f} GB")

# Function to load data from JSON
def load_json(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return []

# Function to load all data
def load_data():
    medicines = load_json('medicines.json')
    symptoms = load_json('symptoms.json')
    diseases = load_json('diseases.json')
    generic_names = load_json('generic_names.json')
    forms = load_json('forms.json')
    brand_names = load_json('brand_names.json')
    manufacturers = load_json('manufacturers.json')
    
    return medicines, symptoms, diseases, generic_names, forms, brand_names, manufacturers

# Merge related data based on foreign keys
def merge_data(medicines, generic_names, brand_names, manufacturers):
    # Convert lists to DataFrames for easier manipulation
    df_medicines = pd.DataFrame(medicines)
    df_generic_names = pd.DataFrame(generic_names)
    df_brand_names = pd.DataFrame(brand_names)
    df_manufacturers = pd.DataFrame(manufacturers)

    # Merge generic names
    if 'generic_name_ids' in df_medicines.columns:
        df_medicines = df_medicines.explode('generic_name_ids')
        df_medicines = df_medicines.merge(df_generic_names, left_on='generic_name_ids', right_on='id', suffixes=('', '_generic')).drop(columns=['generic_name_ids', 'id_generic'])

    # Merge brand names and manufacturers
    if 'brands' in df_medicines.columns:
        df_medicines = df_medicines.explode('brands')
        df_brands = pd.json_normalize(df_medicines['brands'])
        df_medicines = df_medicines.drop(columns=['brands']).reset_index(drop=True)
        df_medicines = df_medicines.join(df_brands)
        df_medicines = df_medicines.merge(df_brand_names, left_on='brand_id', right_on='id', suffixes=('', '_brand')).drop(columns=['brand_id', 'id_brand'])
        df_medicines = df_medicines.merge(df_manufacturers, left_on='manufacturer_id', right_on='id', suffixes=('', '_manufacturer')).drop(columns=['manufacturer_id', 'id_manufacturer'])

    return df_medicines

# Load the pre-trained transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Smaller, efficient model

# Function to create combined text from various fields for retrieval
def create_combined_text(item, fields):
    combined_text = ' '.join([str(item.get(field, '')) for field in fields]).lower()
    return combined_text

# Function to retrieve information
def retrieve_information(data, combined_texts, query, top_k=5):
    try:
        # Encode the query
        query_embedding = model.encode(query.lower(), convert_to_tensor=True)

        # Encode the combined texts
        doc_embeddings = model.encode(combined_texts, convert_to_tensor=True)

        # Compute cosine similarities
        cos_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]

        # Ensure top_k does not exceed the number of available documents
        top_k = min(top_k, len(data))

        # Get the top_k results
        top_results = torch.topk(cos_scores, k=top_k)

        # Retrieve the top_k medicines
        top_medicines = [data.iloc[idx].to_dict() for idx in top_results.indices.tolist()]
        return top_medicines
    except Exception as e:
        st.error(f"An error occurred during information retrieval: {e}")
        return []

# Function to generate response
def generate_response(data, combined_texts, query):
    # Retrieve relevant information based on the query
    relevant_data = retrieve_information(data, combined_texts, query)
    if not relevant_data:
        return "No relevant information found for your query."
    else:
        return relevant_data

# Load the data
medicines, symptoms, diseases, generic_names, forms, brand_names, manufacturers = load_data()

# Merge related data
df_medicines = merge_data(medicines, generic_names, brand_names, manufacturers)

# Detect structure of medicines data
medicine_fields = list(df_medicines.columns)

# Validate data structure
if not medicine_fields:
    st.error("Invalid data format in medicines.json")

# Create combined texts for each medicine
combined_texts = [create_combined_text(row, medicine_fields) for _, row in df_medicines.iterrows()]

# Print initial memory usage
print_memory_usage()

# Title of the Streamlit app
st.title("Medicine Information Retrieval")

# Instructions for the user
st.write("Enter the name or description of the medicine you are looking for in the input box below.")

# Input text box for user query
query = st.text_input("Enter your query about medicine:")

if query:
    # Generate a response based on the query
    relevant_data = generate_response(df_medicines, combined_texts, query)
    
    # Display the response in a structured format
    if relevant_data:
        for item in relevant_data:
            st.subheader(f"Medicine ID: {item.get('id')}")
            st.write(f"**Description**: {item.get('description', 'N/A')}")
            st.write(f"**Description (MM)**: {item.get('description_mm', 'N/A')}")
            st.write(f"**Mechanism of Action**: {item.get('mechanism_of_action', 'N/A')}")
            st.write(f"**Mechanism of Action (MM)**: {item.get('mechanism_of_action_mm', 'N/A')}")
            st.write(f"**Indications**: {', '.join(item.get('indications', []))}")
            st.write(f"**Indications (MM)**: {', '.join(item.get('indications_mm', []))}")
            st.write(f"**Contraindications**: {', '.join(item.get('contraindications', []))}")
            st.write(f"**Contraindications (MM)**: {', '.join(item.get('contraindications_mm', []))}")
            st.write(f"**Warnings**: {', '.join(item.get('warnings', []))}")
            st.write(f"**Warnings (MM)**: {', '.join(item.get('warnings_mm', []))}")
            st.write(f"**Interactions**: {', '.join(item.get('interactions', []))}")
            st.write(f"**Interactions (MM)**: {', '.join(item.get('interactions_mm', []))}")
            st.write(f"**Side Effects**: {', '.join(item.get('side_effects', []))}")
            st.write(f"**Side Effects (MM)**: {', '.join(item.get('side_effects_mm', []))}")
            st.write(f"**Additional Info**: {item.get('additional_info', 'N/A')}")
            st.write(f"**Additional Info (MM)**: {item.get('additional_info_mm', 'N/A')}")
            st.write("---")
    
    # Print memory usage after processing the query
    print_memory_usage()
