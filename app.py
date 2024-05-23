import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import psutil

# Function to print memory usage
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    st.write(f"RSS: {mem_info.rss / (1024 ** 2)} MB, VMS: {mem_info.vms / (1024 ** 2)} MB")

# Function to load data
def load_data():
    medicines = pd.read_json('medicines.json')
    symptoms = pd.read_json('symptoms.json')
    diseases = pd.read_json('diseases.json')
    generic_names = pd.read_json('generic_names.json')
    forms = pd.read_json('forms.json')
    brand_names = pd.read_json('brand_names.json')
    manufacturers = pd.read_json('manufacturers.json')
    
    return medicines, symptoms, diseases, generic_names, forms, brand_names, manufacturers

# Load the pre-trained transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to retrieve information
def retrieve_information(data, query, top_k=5):
    # Check if 'description' column exists
    if 'description' not in data.columns:
        st.error("The 'description' column is missing from the medicines data.")
        return pd.DataFrame()
    
    # Encode the query and the descriptions
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode(data['description'].tolist(), convert_to_tensor=True)

    # Compute cosine similarities
    cos_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]

    # Get the top_k results
    top_results = cos_scores.topk(k=top_k)

    return data.iloc[top_results[1]].reset_index(drop=True)

# Function to generate response
def generate_response(data, query):
    # Retrieve relevant information based on the query
    relevant_data = retrieve_information(data, query)
    if relevant_data.empty:
        return "No relevant information found for your query."
    else:
        response = "Here are the details:\n"
        response += relevant_data.to_string(index=False)
        return response

# Load the data
medicines, symptoms, diseases, generic_names, forms, brand_names, manufacturers = load_data()

# Print initial memory usage
print_memory_usage()

# Title of the Streamlit app
st.title("Medicine Information Retrieval")

# Input text box for user query
query = st.text_input("Enter your query about medicine:")

if query:
    # Generate a response based on the query
    response = generate_response(medicines, query)
    
    # Display the response in the Streamlit app
    st.write(response)
    
    # Print memory usage after processing the query
    print_memory_usage()
