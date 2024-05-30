import streamlit as st
import json
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
def retrieve_graph_data(json_file):
    graph_data = load_graph_data(json_file)
    node_texts = extract_node_data(graph_data)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    
    encoding = tokenize_texts(node_texts, tokenizer)
    embeddings = get_embeddings(encoding, model)
    
    return embeddings, node_texts, graph_data['nodes']

# Function to search for medicines based on query
def search_medicines(query, node_texts, embeddings, tokenizer, model):
    query_encoding = tokenize_texts([query], tokenizer)
    query_embedding = get_embeddings(query_encoding, model)
    
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1]  # Sort indices by similarity in descending order
    
    return top_indices, similarities

# Streamlit UI
st.title('Medicine Data Retrieval using NLP and DistilBERT')

uploaded_file = st.file_uploader("Choose a JSON file", type="json")

if uploaded_file is not None:
    embeddings, node_texts, nodes = retrieve_graph_data(uploaded_file)
    
    st.header('Ask a question about the medicines')
    query = st.text_input('Enter your question:')
    
    if query:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        top_indices, similarities = search_medicines(query, node_texts, embeddings, tokenizer, model)
        
        st.header('Search Results')
        for index in top_indices[:5]:  # Display top 5 results
            st.write(f"**Generic Name:** {nodes[index]['generic_name']}")
            st.write(f"**Commercial Name:** {nodes[index]['commercial_name']}")
            st.write(f"**Description:** {nodes[index]['description']}")
            st.write(f"**Warnings:** {nodes[index]['warnings']}")
            st.write(f"**Dosage:** {nodes[index]['dosage']}")
            st.write(f"**How to use:** {nodes[index]['how_to_use']}")
            st.write(f"**Similarity Score:** {similarities[index]:.4f}")
            st.write("---")
