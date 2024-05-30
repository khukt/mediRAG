import streamlit as st
import json
from transformers import DistilBertTokenizer, DistilBertModel, GPT2Tokenizer, GPT2LMHeadModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load models and tokenizers once and cache them
@st.cache_resource
def load_tokenizer_and_model():
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    return distilbert_tokenizer, distilbert_model, gpt2_tokenizer, gpt2_model

# Load the JSON data
def load_graph_data(json_file):
    data = json.load(json_file)
    return data

# Function to extract and format graph data for transformer processing
def extract_node_data(graph_data):
    node_texts = []
    for node in graph_data['nodes']:
        combined_text = (
            f"Generic Name: {node['generic_name']}. "
            f"Commercial Name: {node['commercial_name']}. "
            f"Description: {node['description']}. "
            f"Warnings: {node['warnings']}. "
            f"Dosage: {node['dosage']}. "
            f"How to use: {node['how_to_use']}."
        )
        node_texts.append(combined_text)
    return node_texts

# Tokenize and encode the node texts
def tokenize_texts(texts, tokenizer):
    encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    return encoding

# Get the embeddings from the transformer model
def get_embeddings(encoding, model):
    with torch.no_grad():
        outputs = model(**encoding)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Use mean pooling to get fixed-size embeddings
    return embeddings

# Main function to retrieve data
def retrieve_graph_data(json_file, tokenizer, model):
    graph_data = load_graph_data(json_file)
    node_texts = extract_node_data(graph_data)
    
    encoding = tokenize_texts(node_texts, tokenizer)
    embeddings = get_embeddings(encoding, model)
    
    return embeddings, node_texts, graph_data['nodes']

# Function to search for medicines based on query
def search_medicines(query, node_texts, embeddings, tokenizer, model):
    query_encoding = tokenize_texts([query], tokenizer)
    query_embedding = get_embeddings(query_encoding, model)
    
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_index = np.argmax(similarities)  # Get the index of the highest similarity
    
    return top_index, similarities[top_index]

# Generate a formal explanatory paragraph using retrieved data
def generate_explanation(node):
    explanation = (
        f"{node['commercial_name']} (generic name: {node['generic_name']}) is used to treat {node['description'].lower()}. "
        f"The recommended dosage is {node['dosage']}. {node['warnings']} "
        f"To use this medication, {node['how_to_use'].lower()}"
    )
    return explanation

# Streamlit UI
st.title('Medicine Data Retrieval and Explanation using NLP and DistilBERT')

uploaded_file = st.file_uploader("Choose a JSON file", type="json")

if uploaded_file is not None:
    distilbert_tokenizer, distilbert_model, gpt2_tokenizer, gpt2_model = load_tokenizer_and_model()
    embeddings, node_texts, nodes = retrieve_graph_data(uploaded_file, distilbert_tokenizer, distilbert_model)
    
    st.header('Ask a question about the medicines')
    query = st.text_input('Enter your question:')
    
    if query:
        top_index, similarity = search_medicines(query, node_texts, embeddings, distilbert_tokenizer, distilbert_model)
        
        st.header('Search Result')
        
        node = nodes[top_index]
        text = (
            f"Generic Name: {node['generic_name']}. "
            f"Commercial Name: {node['commercial_name']}. "
            f"Description: {node['description']}. "
            f"Warnings: {node['warnings']}. "
            f"Dosage: {node['dosage']}. "
            f"How to use: {node['how_to_use']}."
        )
        st.write(f"**Generic Name:** {node['generic_name']}")
        st.write(f"**Commercial Name:** {node['commercial_name']}")
        st.write(f"**Description:** {node['description']}")
        st.write(f"**Warnings:** {node['warnings']}")
        st.write(f"**Dosage:** {node['dosage']}")
        st.write(f"**How to use:** {node['how_to_use']}")
        st.write(f"**Similarity Score:** {similarity:.4f}")
        
        # Generate formal explanation
        explanation = generate_explanation(node)
        st.write("**Explanation:**")
        st.write(explanation)
        st.write("---")
