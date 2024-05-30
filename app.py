import streamlit as st
from transformers import pipeline
import sqlite3
import json

# Create and populate SQLite database
def create_and_populate_db():
    conn = sqlite3.connect(':memory:')  # Use in-memory database for simplicity
    c = conn.cursor()

    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS medicines (
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

    # Sample data
    medicines_data = [
        {
            "generic_name": "Paracetamol",
            "brand_names": ["Tylenol", "Panadol", "Acetaminophen"],
            "description": "Paracetamol is a medication used to treat pain and fever. It is commonly used for headaches, muscle aches, arthritis, backaches, toothaches, colds, and fevers.",
            "dosage_forms": [
                {
                    "form": "tablet",
                    "strengths": ["500mg", "650mg"]
                },
                {
                    "form": "syrup",
                    "strengths": ["120mg/5ml"]
                }
            ],
            "indications": [
                "Headache",
                "Muscle ache",
                "Fever",
                "Arthritis"
            ],
            "contraindications": [
                "Severe liver disease",
                "Allergic reaction to paracetamol"
            ],
            "side_effects": {
                "common": ["Nausea", "Vomiting"],
                "serious": ["Liver damage", "Severe allergic reaction"]
            },
            "interactions": [
                {
                    "drug": "Warfarin",
                    "description": "Paracetamol can increase the blood-thinning effect of Warfarin, increasing the risk of bleeding."
                },
                {
                    "drug": "Alcohol",
                    "description": "Concurrent use can increase the risk of liver damage."
                }
            ],
            "warnings": [
                "Do not exceed the recommended dose.",
                "Use with caution in patients with liver disease."
            ],
            "mechanism_of_action": "Paracetamol works by inhibiting the synthesis of prostaglandins, which help transmit pain and induce fever.",
            "pharmacokinetics": {
                "absorption": "Rapidly absorbed from the gastrointestinal tract.",
                "metabolism": "Metabolized in the liver.",
                "half_life": "1-4 hours",
                "excretion": "Excreted by the kidneys."
            },
            "patient_information": [
                "Take paracetamol with or without food.",
                "Do not take more than 4 grams (4000 mg) in 24 hours."
            ]
        },
        {
            "generic_name": "Ibuprofen",
            "brand_names": ["Advil", "Motrin", "Nurofen"],
            "description": "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce fever and treat pain or inflammation caused by many conditions such as headache, toothache, arthritis, menstrual cramps, or minor injury.",
            "dosage_forms": [
                {
                    "form": "tablet",
                    "strengths": ["200mg", "400mg", "600mg", "800mg"]
                },
                {
                    "form": "suspension",
                    "strengths": ["100mg/5ml"]
                }
            ],
            "indications": [
                "Pain",
                "Fever",
                "Inflammation"
            ],
            "contraindications": [
                "History of asthma or allergic reaction to aspirin or other NSAIDs",
                "Active gastrointestinal bleeding"
            ],
            "side_effects": {
                "common": ["Nausea", "Heartburn", "Dizziness"],
                "serious": ["Gastrointestinal bleeding", "Kidney damage"]
            },
            "interactions": [
                {
                    "drug": "Aspirin",
                    "description": "Concurrent use can increase the risk of gastrointestinal bleeding."
                },
                {
                    "drug": "Warfarin",
                    "description": "Ibuprofen can enhance the anticoagulant effect of Warfarin."
                }
            ],
            "warnings": [
                "Do not take more than the recommended dose.",
                "Use with caution in patients with a history of heart disease or gastrointestinal issues."
            ],
            "mechanism_of_action": "Ibuprofen works by inhibiting the enzymes COX-1 and COX-2, which are involved in the synthesis of prostaglandins that mediate inflammation, pain, and fever.",
            "pharmacokinetics": {
                "absorption": "Well absorbed from the gastrointestinal tract.",
                "metabolism": "Metabolized in the liver.",
                "half_life": "2-4 hours",
                "excretion": "Excreted by the kidneys."
            },
            "patient_information": [
                "Take ibuprofen with food or milk to reduce stomach upset.",
                "Avoid alcohol while taking this medication."
            ]
        }
    ]

    # Insert data into the database
    for med in medicines_data:
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

    conn.commit()
    return conn

conn = create_and_populate_db()
c = conn.cursor()

# Load the QA model
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

st.title("Medicines Information System")

# User input
question = st.text_input("Ask a question about any medicine:")

def fetch_medicine_data(keyword):
    c.execute('SELECT * FROM medicines WHERE generic_name LIKE ? OR brand_names LIKE ?', (f'%{keyword}%', f'%{keyword}%'))
    rows = c.fetchall()
    if rows:
        return [{
            "generic_name": row[0],
            "brand_names": json.loads(row[1]),
            "description": row[2],
            "indications": json.loads(row[3]),
            "contraindications": json.loads(row[4]),
            "common_side_effects": json.loads(row[5]),
            "serious_side_effects": json.loads(row[6]),
            "interactions": json.loads(row[7]),
            "warnings": json.loads(row[8]),
            "mechanism_of_action": row[9],
            "pharmacokinetics": json.loads(row[10]),
            "patient_information": json.loads(row[11])
        } for row in rows]
    return None

def build_relevant_context(medicines):
    context = ""
    for med in medicines:
        context += f"Generic Name: {med['generic_name']}\n"
        context += f"Brand Names: {', '.join(med['brand_names'])}\n"
        context += f"Description: {med['description']}\n"
        context += f"Indications: {', '.join(med['indications'])}\n"
        context += f"Contraindications: {', '.join(med['contraindications'])}\n"
        context += f"Common Side Effects: {', '.join(med['common_side_effects'])}\n"
        context += f"Serious Side Effects: {', '.join(med['serious_side_effects'])}\n"
        context += f"Interactions: {', '.join([i['drug'] + ': ' + i['description'] for i in med['interactions']])}\n"
        context += f"Warnings: {', '.join(med['warnings'])}\n"
        context += f"Mechanism of Action: {med['mechanism_of_action']}\n"
        pk = med['pharmacokinetics']
        context += f"Pharmacokinetics: Absorption: {pk['absorption']}; Metabolism: {pk['metabolism']}; Half-life: {pk['half_life']}; Excretion: {pk['excretion']}\n"
        context += f"Patient Information: {', '.join(med['patient_information'])}\n"
    return context

if question:
    keywords = question.lower().split()
    relevant_meds = []
    for keyword in keywords:
        meds = fetch_medicine_data(keyword)
        if meds:
            relevant_meds.extend(meds)
    
    if relevant_meds:
        context = build_relevant_context(relevant_meds)
        
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
    keywords = test["question"].lower().split()
    relevant_meds = []
    for keyword in keywords:
        meds = fetch_medicine_data(keyword)
        if meds:
            relevant_meds.extend(meds)
    
    if relevant_meds:
        context = build_relevant_context(relevant_meds)
        
        if context:
            answer = qa_pipeline(question=test["question"], context=context)
            st.write(f"**Question:** {test['question']}")
            st.write(f"**Expected Answer:** {test['expected']}")
            st.write(f"**Model's Answer:** {answer['answer']}")
        else:
            st.write(f"**Question:** {test['question']}")
            st.write(f"**Expected Answer:** {test['expected']}")
            st.write("**Model's Answer:** No relevant context found for the question.")
    else:
        st.write(f"**Question:** {test['question']}")
        st.write(f"**Expected Answer:** {test['expected']}")
        st.write("**Model's Answer:** No relevant context found for the question.")
    st.write("---")
