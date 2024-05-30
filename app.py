import streamlit as st
from transformers import AlbertTokenizer, AlbertForQuestionAnswering, pipeline
import json

# Load the medicines data from the JSON file
with open('medicines.json', 'r') as f:
    medicines = json.load(f)



tokenizer = AlbertTokenizer.from_pretrained('ahotrod/albert_xxlargev1_squad2_512')
model = AlbertForQuestionAnswering.from_pretrained('ahotrod/albert_xxlargev1_squad2_512')
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)


st.title("Medicines Information System")

# User input
question = st.text_input("Ask a question about any medicine:")

def build_relevant_context(question, medicines):
    context = ""
    keywords = question.lower().split()
    for drug in medicines:
        drug_info = []
        if any(kw in drug['generic_name'].lower() for kw in keywords) or any(kw.lower() in [bn.lower() for bn in drug['brand_names']] for kw in keywords):
            drug_info.append(f"Generic Name: {drug['generic_name']}\n")
            drug_info.append(f"Brand Names: {', '.join(drug['brand_names'])}\n")
            drug_info.append(f"Description: {drug['description']}\n")
            dosage_forms = ", ".join([f"{d['form']} ({', '.join(d['strengths'])})" for d in drug['dosage_forms']])
            drug_info.append(f"Dosage Forms: {dosage_forms}\n")
            drug_info.append(f"Indications: {', '.join(drug['indications'])}\n")
            drug_info.append(f"Contraindications: {', '.join(drug['contraindications'])}\n")
            drug_info.append("Side Effects: Common: " + ", ".join(drug['side_effects']['common']) + "; Serious: " + ", ".join(drug['side_effects']['serious']) + "\n")
            interactions = "; ".join([f"{i['drug']}: {i['description']}" for i in drug['interactions']])
            drug_info.append(f"Interactions: {interactions}\n")
            drug_info.append(f"Warnings: {', '.join(drug['warnings'])}\n")
            drug_info.append(f"Mechanism of Action: {drug['mechanism_of_action']}\n")
            pharmacokinetics = f"Absorption: {drug['pharmacokinetics']['absorption']}; Metabolism: {drug['pharmacokinetics']['metabolism']}; Half-life: {drug['pharmacokinetics']['half_life']}; Excretion: {drug['pharmacokinetics']['excretion']}"
            drug_info.append(f"Pharmacokinetics: {pharmacokinetics}\n")
            drug_info.append(f"Patient Information: {', '.join(drug['patient_information'])}\n")
            context += " ".join(drug_info)
    return context

if question:
    # Build the context relevant to the question
    context = build_relevant_context(question, medicines)
    if context:
        # Summarize the context
        summarized_context = summarization_pipeline(context, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        
        # Get the answer from the QA model using the summarized context
        answer = qa_pipeline(question=question, context=summarized_context)
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
        summarized_context = summarization_pipeline(context, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        answer = qa_pipeline(question=test["question"], context=summarized_context)
        st.write(f"**Question:** {test['question']}")
        st.write(f"**Expected Answer:** {test['expected']}")
        st.write(f"**Model's Answer:** {answer['answer']}")
    else:
        st.write(f"**Question:** {test['question']}")
        st.write(f"**Expected Answer:** {test['expected']}")
        st.write("**Model's Answer:** No relevant context found for the question.")
    st.write("---")
