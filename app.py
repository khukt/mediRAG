import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
import psutil
import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

# Function to classify the type of query
def classify_query(query):
    symptoms_keywords = ['headache', 'fever', 'cough', 'pain', 'symptoms', 'ache']
    medicine_keywords = ['medicine', 'treatment', 'medication']
    comparison_keywords = ['difference', 'compare', 'versus', 'vs']

    if any(keyword in query.lower() for keyword in symptoms_keywords):
        return 'symptom'
    elif any(keyword in query.lower() for keyword in comparison_keywords):
        return 'comparison'
    else:
        return 'medicine'

# Function to handle symptom queries
def handle_symptom_query(query, symptoms_df):
    vectorizer = CountVectorizer().fit_transform(symptoms_df['description'].tolist())
    vectors = vectorizer.toarray()
    query_vector = vectorizer.transform([query]).toarray()
    cosine_similarities = cosine_similarity(query_vector, vectors).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-5:-1]
    return symptoms_df.iloc[related_docs_indices]

# Function to handle comparison queries
def handle_comparison_query(query, medicines_df):
    # Extract medicine names from the query
    medicines_in_query = [name for name in medicines_df['name'].tolist() if name.lower() in query.lower()]
    if len(medicines_in_query) < 2:
        return "Please specify two medicines to compare."

    # Retrieve information for the specified medicines
    results = medicines_df[medicines_df['name'].str.lower().isin([med.lower() for med in medicines_in_query])]
    return results

# Function to handle general medicine queries
def handle_medicine_query(query, combined_texts, medicines_df):
    try:
        # Encode the query
        query_embedding = model.encode(query.lower(), convert_to_tensor=True)

        # Encode the combined texts
        doc_embeddings = model.encode(combined_texts, convert_to_tensor=True)

        # Compute cosine similarities
        cos_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]

        # Ensure top_k does not exceed the number of available documents
        top_k = min(5, len(medicines_df))

        # Get the top_k results
        top_results = torch.topk(cos_scores, k=top_k)

        # Retrieve the top_k medicines
        top_medicines = [medicines_df.iloc[idx].to_dict() for idx in top_results.indices.tolist()]
        return top_medicines
    except Exception as e:
        st.error(f"An error occurred during information retrieval: {e}")
        return []

# Function to generate response based on query type
def generate_response(query, medicines_df, symptoms_df, combined_texts):
    query_type = classify_query(query)
    
    if query_type == 'symptom':
        relevant_data = handle_symptom_query(query, symptoms_df)
        response = "Based on your symptoms, here are some related medicines:\n"
        for item in relevant_data.to_dict(orient='records'):
            response += json.dumps(item, indent=2, ensure_ascii=False) + "\n"
        return response
    elif query_type == 'comparison':
        relevant_data = handle_comparison_query(query, medicines_df)
        if isinstance(relevant_data, str):
            return relevant_data
        response = "Here is the comparison between the medicines:\n"
        for item in relevant_data.to_dict(orient='records'):
            response += json.dumps(item, indent=2, ensure_ascii=False) + "\n"
        return response
    else:
        relevant_data = handle_medicine_query(query, combined_texts, medicines_df)
        response = "Here are the details about the medicine:\n"
        for item in relevant_data:
            response += json.dumps(item, indent=2, ensure_ascii=False) + "\n"
        return response

# Load the data
medicines, symptoms, diseases, generic_names, forms, brand_names, manufacturers = load_data()

# Merge related data
df_medicines = merge_data(medicines, generic_names, brand_names, manufacturers)
df_symptoms = pd.DataFrame(symptoms)

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
st.write("Enter your query about medicine in the input box below. You can ask about symptoms, specific medicines, or compare different medicines.")

# Input text box for user query
query = st.text_input("Enter your query about medicine:")

if query:
    # Generate a response based on the query
    response = generate_response(query, df_medicines, df_symptoms, combined_texts)
    
    # Display the response in the Streamlit app
    st.write(response)
    
    # Print memory usage after processing the query
    print_memory_usage()
