import sqlite3
import json

# Connect to SQLite database (or create it)
conn = sqlite3.connect('medicines.db')
c = conn.cursor()

# Create tables
c.execute('''CREATE TABLE IF NOT EXISTS medicines (
    id INTEGER PRIMARY KEY,
    generic_name TEXT,
    brand_names TEXT,
    description TEXT,
    indications TEXT,
    contraindications TEXT,
    common_side_effects TEXT,
    serious_side_effects TEXT,
    interactions TEXT,
    warnings TEXT,
    mechanism_of_action TEXT,
    pharmacokinetics TEXT,
    patient_information TEXT
)''')

# Load data from JSON file
with open('medicines.json', 'r') as f:
    medicines = json.load(f)

# Insert data into the database
for med in medicines:
    c.execute('''INSERT INTO medicines (generic_name, brand_names, description, indications, contraindications, common_side_effects, serious_side_effects, interactions, warnings, mechanism_of_action, pharmacokinetics, patient_information)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
        med['generic_name'],
        json.dumps(med['brand_names']),
        med['description'],
        json.dumps(med['indications']),
        json.dumps(med['contraindications']),
        json.dumps(med['side_effects']['common']),
        json.dumps(med['side_effects']['serious']),
        json.dumps(med['interactions']),
        json.dumps(med['warnings']),
        med['mechanism_of_action'],
        json.dumps(med['pharmacokinetics']),
        json.dumps(med['patient_information'])
    ))

# Commit changes and close the connection
conn.commit()
conn.close()


import streamlit as st
from transformers import pipeline
import sqlite3
import json

# Load the QA model
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

# Connect to SQLite database
conn = sqlite3.connect('medicines.db')
c = conn.cursor()

st.title("Medicines Information System")

# User input
question = st.text_input("Ask a question about any medicine:")

def fetch_medicine_data(generic_name):
    c.execute('SELECT * FROM medicines WHERE generic_name = ?', (generic_name,))
    return c.fetchone()

def build_relevant_context(question, medicines):
    context = ""
    keywords = question.lower().split()
    for med in medicines:
        if any(kw in med['generic_name'].lower() for kw in keywords) or any(kw.lower() in med['brand_names'].lower() for kw in keywords):
            context += f"Generic Name: {med['generic_name']}\n"
            context += f"Brand Names: {', '.join(json.loads(med['brand_names']))}\n"
            context += f"Description: {med['description']}\n"
            context += f"Indications: {', '.join(json.loads(med['indications']))}\n"
            context += f"Contraindications: {', '.join(json.loads(med['contraindications']))}\n"
            context += f"Common Side Effects: {', '.join(json.loads(med['common_side_effects']))}\n"
            context += f"Serious Side Effects: {', '.join(json.loads(med['serious_side_effects']))}\n"
            context += f"Interactions: {', '.join([i['drug'] + ': ' + i['description'] for i in json.loads(med['interactions'])])}\n"
            context += f"Warnings: {', '.join(json.loads(med['warnings']))}\n"
            context += f"Mechanism of Action: {med['mechanism_of_action']}\n"
            pk = json.loads(med['pharmacokinetics'])
            context += f"Pharmacokinetics: Absorption: {pk['absorption']}; Metabolism: {pk['metabolism']}; Half-life: {pk['half_life']}; Excretion: {pk['excretion']}\n"
            context += f"Patient Information: {', '.join(json.loads(med['patient_information']))}\n"
    return context

if question:
    # Extract the relevant generic name from the question
    keywords = question.lower().split()
    generic_names = [med['generic_name'] for med in medicines if any(kw in med['generic_name'].lower() for kw in keywords)]
    
    if generic_names:
        generic_name = generic_names[0]  # Assuming the first match is the correct one
        med_data = fetch_medicine_data(generic_name)
        context = build_relevant_context(question, [med_data])
        
        if context:
            # Get the answer from the QA model
            answer = qa_pipeline(question=question, context=context)
            st.write("Answer:", answer['answer'])
        else:
            st.write("No relevant context found for the question.")
    else:
        st.write("No relevant medicine found for the question.")

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
    generic_name = test["question"].split()[2].lower()  # Extract the generic name from the question
    med_data = fetch_medicine_data(generic_name)
    context = build_relevant_context(test["question"], [med_data])
    
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
