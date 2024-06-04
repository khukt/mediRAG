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


def build_relevant_context(question, medicines):
    context = ""
    keywords = question.lower().split()
    for drug in medicines:
        if any(kw in drug['generic_name'].lower() for kw in keywords) or any(kw.lower() in [bn.lower() for bn in drug['brand_names']] for kw in keywords):
            context += f"Generic Name: {drug['generic_name']}\n"
            context += f"Brand Names: {', '.join(drug['brand_names'])}\n"
            context += f"Description: {drug['description']}\n"
            context += f"Indications: {', '.join(drug['indications'])}\n"
            context += f"Contraindications: {', '.join(drug['contraindications'])}\n"
            context += "Side Effects: Common: " + ", ".join(drug['side_effects']['common']) + "; Serious: " + ", ".join(drug['side_effects']['serious']) + "\n"
            context += "; ".join([f"Interactions: {i['drug']}: {i['description']}" for i in drug['interactions']]) + "\n"
            context += f"Warnings: {', '.join(drug['warnings'])}\n"
            context += f"Mechanism of Action: {drug['mechanism_of_action']}\n"
            pharmacokinetics = f"Pharmacokinetics: Absorption: {drug['pharmacokinetics']['absorption']}; Metabolism: {drug['pharmacokinetics']['metabolism']}; Half-life: {drug['pharmacokinetics']['half_life']}; Excretion: {drug['pharmacokinetics']['excretion']}"
            context += f"{pharmacokinetics}\n"
            context += f"Patient Information: {', '.join(drug['patient_information'])}\n"
    return context

if question:
    # Build the context relevant to the question
    context = build_relevant_context(question, medicines)
    if context:
        input_text = f"Question: {question}\nContext: {context}\nAnswer:"
        # Generate the answer from the GPT-2 model
        answer = qa_pipeline(input_text, max_length=200, num_return_sequences=1)
        st.write("Answer:", answer[0]['generated_text'].split('Answer:')[1].strip())
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
        input_text = f"Question: {test['question']}\nContext: {context}\nAnswer:"
        answer = qa_pipeline(input_text, max_length=200, num_return_sequences=1)
        st.write(f"**Question:** {test['question']}")
        st.write(f"**Expected Answer:** {test['expected']}")
        st.write(f"**Model's Answer:** {answer[0]['generated_text'].split('Answer:')[1].strip()}")
    else:
        st.write(f"**Question:** {test['question']}")
        st.write(f"**Expected Answer:** {test['expected']}")
        st.write("**Model's Answer:** No relevant context found for the question.")
    st.write("---")
