from transformers import AlbertTokenizer, AlbertForQuestionAnswering, pipeline
import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the medicines data from the JSON file
@st.cache_resource
def load_medicines():
    with open('medicines.json', 'r') as f:
        return json.load(f)

# Load the ALBERT model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AlbertTokenizer.from_pretrained('twmkn9/albert-base-v2-squad2')
    model = AlbertForQuestionAnswering.from_pretrained('twmkn9/albert-base-v2-squad2')
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline

# Extract relevant data sections for better context extraction
def extract_relevant_sections(medicines):
    sections = []
    for drug in medicines:
        for key, value in drug.items():
            if isinstance(value, list):
                for item in value:
                    sections.append(f"{key}: {', '.join(item) if isinstance(item, list) else item}")
            else:
                sections.append(f"{key}: {value}")
    return sections

# Function to build relevant context using TF-IDF and cosine similarity
def build_relevant_context(question, sections):
    vectorizer = TfidfVectorizer().fit_transform(sections)
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, vectorizer).flatten()
    relevant_sections = [sections[i] for i in similarities.argsort()[-5:][::-1]]
    return " ".join(relevant_sections)

medicines = load_medicines()
qa_pipeline = load_model()
sections = extract_relevant_sections(medicines)

st.title("Medicines Information System")

# User input
question = st.text_input("Ask a question about any medicine:")

if question:
    context = build_relevant_context(question, sections)
    if context:
        try:
            # Get the answer from the QA model
            answer = qa_pipeline(question=question, context=context)
            st.write("Answer:", answer['answer'])
        except Exception as e:
            st.error(f"An error occurred: {e}")
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
    context = build_relevant_context(test["question"], sections)
    if context:
        try:
            answer = qa_pipeline(question=test["question"], context=context)
            st.write(f"**Question:** {test['question']}")
            st.write(f"**Expected Answer:** {test['expected']}")
            st.write(f"**Model's Answer:** {answer['answer']}")
        except Exception as e:
            st.write(f"**Question:** {test['question']}")
            st.write(f"**Expected Answer:** {test['expected']}")
            st.write(f"**Model's Answer:** An error occurred: {e}")
    else:
        st.write(f"**Question:** {test['question']}")
        st.write(f"**Expected Answer:** {test['expected']}")
        st.write("**Model's Answer:** No relevant context found for the question.")
    st.write("---")
