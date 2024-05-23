import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
import psutil
import torch

# Function to print memory usage in GB
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    st.write(f"RSS: {mem_info.rss / (1024 ** 3):.2f} GB, VMS: {mem_info.vms / (1024 ** 3):.2f} GB")

# Function to load data from JSON
def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Function to load data
def load_data():
    medicines = load_json('medicines.json')
    symptoms = load_json('symptoms.json')
    diseases = load_json('diseases.json')
    generic_names = load_json('generic_names.json')
    forms = load_json('forms.json')
    brand_names = load_json('brand_names.json')
    manufacturers = load_json('manufacturers.json')
    
    return medicines, symptoms, diseases, generic_names, forms, brand_names, manufacturers

# Load the pre-trained transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Smaller, efficient model

# Function to create combined text from various fields for retrieval
def create_combined_text(item):
    fields = [
        'description', 'mechanism_of_action', 'indications', 
        'contraindications', 'warnings', 'interactions', 
        'side_effects', 'additional_info'
    ]
    combined_text = ' '.join([item.get(field, '') for field in fields]).lower()
    return combined_text

# Function to retrieve information
def retrieve_information(data, query, top_k=5):
    try:
        # Encode the query
        query_embedding = model.encode(query.lower(), convert_to_tensor=True)

        combined_texts = [create_combined_text(item) for item in data]
        
        # Encode the combined texts
        doc_embeddings = model.encode(combined_texts, convert_to_tensor=True)

        # Compute cosine similarities
        cos_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]

        # Ensure top_k does not exceed the number of available documents
        top_k = min(top_k, len(data))

        # Get the top_k results
        top_results = torch.topk(cos_scores, k=top_k)

        # Retrieve the top_k medicines
        top_medicines = [data[idx] for idx in top_results.indices.tolist()]
        return top_medicines
    except Exception as e:
        st.error(f"An error occurred during information retrieval: {e}")
        return []

# Function to generate response
def generate_response(data, query):
    # Retrieve relevant information based on the query
    relevant_data = retrieve_information(data, query)
    if not relevant_data:
        return "No relevant information found for your query."
    else:
        response = "Here are the details:\n"
        for item in relevant_data:
            response += json.dumps(item, indent=2) + "\n"
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
