import streamlit as st
from transformers import pipeline
import json

# Load the medicines data from the JSON file
with open('medicines.json', 'r') as f:
    medicines = json.load(f)

# Load a lightweight transformer model for question answering
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

st.title("Medicines Information System")

# User input
question = st.text_input("Ask a question about any medicine:")

def build_context(drug):
    context = ""
    for key, value in drug.items():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for sub_key, sub_value in item.items():
                        context += str(sub_value) + " "
                else:
                    context += str(item) + " "
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                context += str(sub_value) + " "
        else:
            context += str(value) + " "
    return context

if question:
    # Search for relevant drug information
    context = ""
    for drug in medicines:
        context += build_context(drug)

    # Get the answer from the QA model
    answer = qa_pipeline(question=question, context=context)
    st.write("Answer:", answer['answer'])
