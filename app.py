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

def build_context(medicines):
    context = ""
    for drug in medicines:
        context += f"Generic Name: {drug['generic_name']}\n"
        context += f"Brand Names: {', '.join(drug['brand_names'])}\n"
        context += f"Description: {drug['description']}\n"
        dosage_forms = ", ".join([f"{d['form']} ({', '.join(d['strengths'])})" for d in drug['dosage_forms']])
        context += f"Dosage Forms: {dosage_forms}\n"
        context += f"Indications: {', '.join(drug['indications'])}\n"
        context += f"Contraindications: {', '.join(drug['contraindications'])}\n"
        context += "Side Effects: Common: " + ", ".join(drug['side_effects']['common']) + "; Serious: " + ", ".join(drug['side_effects']['serious']) + "\n"
        interactions = "; ".join([f"{i['drug']}: {i['description']}" for i in drug['interactions']])
        context += f"Interactions: {interactions}\n"
        context += f"Warnings: {', '.join(drug['warnings'])}\n"
        context += f"Mechanism of Action: {drug['mechanism_of_action']}\n"
        pharmacokinetics = f"Absorption: {drug['pharmacokinetics']['absorption']}; Metabolism: {drug['pharmacokinetics']['metabolism']}; Half-life: {drug['pharmacokinetics']['half_life']}; Excretion: {drug['pharmacokinetics']['excretion']}"
        context += f"Pharmacokinetics: {pharmacokinetics}\n"
        context += f"Patient Information: {', '.join(drug['patient_information'])}\n"
    return context

if question:
    # Build the context from the JSON data
    context = build_context(medicines)

    # Get the answer from the QA model
    answer = qa_pipeline(question=question, context=context)
    st.write("Answer:", answer['answer'])
