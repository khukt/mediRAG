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

def extract_relevant_info(drug, info_type):
    if info_type == "usage":
        return drug.get("description", "")
    elif info_type == "brand_names":
        return ", ".join(drug.get("brand_names", []))
    elif info_type == "side_effects":
        common = ", ".join(drug.get("side_effects", {}).get("common", []))
        serious = ", ".join(drug.get("side_effects", {}).get("serious", []))
        return f"Common: {common}; Serious: {serious}"
    elif info_type == "contraindications":
        return ", ".join(drug.get("contraindications", []))
    elif info_type == "mechanism_of_action":
        return drug.get("mechanism_of_action", "")
    elif info_type == "dosage":
        return ", ".join(drug.get("patient_information", []))
    return ""

def build_relevant_context(question, medicines):
    context = ""
    keywords = question.lower().split()
    info_type = ""
    
    if "used for" in question or "usage" in question:
        info_type = "usage"
    elif "brand names" in question:
        info_type = "brand_names"
    elif "side effects" in question:
        info_type = "side_effects"
    elif "contraindications" in question:
        info_type = "contraindications"
    elif "mechanism of action" in question:
        info_type = "mechanism_of_action"
    elif "how should i take" in question or "dosage" in question:
        info_type = "dosage"
    
    for drug in medicines:
        if any(kw in drug['generic_name'].lower() for kw in keywords) or any(kw.lower() in [bn.lower() for bn in drug['brand_names']] for kw in keywords):
            context += extract_relevant_info(drug, info_type) + " "
    
    return context.strip()

if question:
    # Build the context relevant to the question
    context = build_relevant_context(question, medicines)
    if context:
        # Get the answer from the QA model
        answer = qa_pipeline(question=question, context=context)
        st.write("Answer:", answer['answer'])
    else:
        st.write("No relevant context found for the question.")

# Predefined test questions and expected answers
test_questions = [
    {"question": "What is Paracetamol used for?", "expected": "Paracetamol is a medication used to treat pain and fever. It is commonly used for headaches, muscle aches, arthritis, backaches, toothaches, colds, and fevers."},
    {"question": "What are the brand names for Ibuprofen?", "expected": "Advil, Motrin, Nurofen"},
    {"question": "What are the side effects of Paracetamol?", "expected": "Common side effects include nausea and vomiting. Serious side effects include liver damage and severe allergic reactions."},
    {"question": "What are the contraindications for Ibuprofen?", "expected": "History of asthma or allergic reaction to aspirin or other NSAIDs, active gastrointestinal bleeding."},
    {"question": "What is the mechanism of action of Ibuprofen?", "expected": "Ibuprofen works by inhibiting the enzymes COX-1 and COX-2, which are involved in the synthesis of prostaglandins that mediate inflammation, pain, and fever."},
    {"question": "How should I take Paracetamol?", "expected": "Take paracetamol with or without food. Do not take more than 4 grams (4000 mg) in 24 hours."}
]

st.subheader("Test Questions and Expected Answers")

for test in test_questions:
    context = build_relevant_context(test["question"], medicines)
    if context:
        answer = qa_pipeline(question=test["question"], context=context)
        st.write(f"**Question:** {test['question']}")
        st.write(f"**Expected Answer:** {test['expected']}")
        st.write(f"**Model's Answer:** {answer['answer']}")
    else:
        st.write(f"**Question:** {test['question']}")
        st.write(f"**Expected Answer:** {test['expected']}")
        st.write("**Model's Answer:** No relevant context found for the question.")
    st.write("---")
