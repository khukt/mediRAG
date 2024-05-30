import streamlit as st
import json
from transformers import DistilBertTokenizer, DistilBertModel, DistilGPT2Tokenizer, DistilGPT2LMHeadModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load models and tokenizers once and cache them
@st.cache_resource
def load_tokenizer_and_model():
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    distilgpt2_tokenizer = DistilGPT2Tokenizer.from_pretrained('distilgpt2')
    distilgpt2_model = DistilGPT2LMHeadModel.from_pretrained('distilgpt2')
    return distilbert_tokenizer, distilbert_model, distilgpt2_tokenizer, distilgpt2_model

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
    top_indices = np.argsort(similarities)[::-1]  # Sort indices by similarity in descending order
    
    return top_indices, similarities

# Generate a summary or explanation using DistilGPT2
def generate_summary(text, tokenizer, model):
    inputs = tokenizer.encode("Summarize: " + text, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Streamlit UI
st.title('Medicine Data Retrieval and Summarization using NLP and DistilBERT')

uploaded_file = st.file_uploader("Choose a JSON file", type="json")

if uploaded_file is not None:
    distilbert_tokenizer, distilbert_model, distilgpt2_tokenizer, distilgpt2_model = load_tokenizer_and_model()
    embeddings, node_texts, nodes = retrieve_graph_data(uploaded_file, distilbert_tokenizer, distilbert_model)
    
    st.header('Ask a question about the medicines')
    query = st.text_input('Enter your question:')
    
    if query:
        top_indices, similarities = search_medicines(query, node_texts, embeddings, distilbert_tokenizer, distilbert_model)
        
        st.header('Search Results')
        
        for index in top_indices[:3]:  # Display top 3 results and generate summaries
            text = (
                f"Generic Name: {nodes[index]['generic_name']}. "
                f"Commercial Name: {nodes[index]['commercial_name']}. "
                f"Description: {nodes[index]['description']}. "
                f"Warnings: {nodes[index]['warnings']}. "
                f"Dosage: {nodes[index]['dosage']}. "
                f"How to use: {nodes[index]['how_to_use']}."
            )
            st.write(f"**Generic Name:** {nodes[index]['generic_name']}")
            st.write(f"**Commercial Name:** {nodes[index]['commercial_name']}")
            st.write(f"**Description:** {nodes[index]['description']}")
            st.write(f"**Warnings:** {nodes[index]['warnings']}")
            st.write(f"**Dosage:** {nodes[index]['dosage']}")
            st.write(f"**How to use:** {nodes[index]['how_to_use']}")
            st.write(f"**Similarity Score:** {similarities[index]:.4f}")
            
            # Generate summary
            summary = generate_summary(text, distilgpt2_tokenizer, distilgpt2_model)
            st.write("**Summary:**")
            st.write(summary)
            st.write("---")

