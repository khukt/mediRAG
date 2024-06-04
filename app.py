from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import streamlit as st
import json

# Load the medicines data from the JSON file
with open('medicines.json', 'r') as f:
    medicines = json.load(f)

# Load the GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

st.title("Medicines Information System")

# User input
question = st.text_input("Ask a question about any medicine:")
