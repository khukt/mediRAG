from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator

# Load the medicines data from the JSON file
@st.cache_resource
def load_medicines():
    with open('medicines.json', 'r') as f:
        return json.load(f)

# Load the multilingual XLM-RoBERTa model and tokenizer for QA
@st.cache_resource
def load_qa_model():
    tokenizer = AutoTokenizer.from_pretrained('deepset/xlm-roberta-base-squad2')
    model = AutoModelForQuestionAnswering.from_pretrained('deepset/xlm-roberta-base-squad2')
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline

# Load the Sentence Transformer model for semantic search
@st.cache_resource
def load_sentence_transformer_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Initialize the Google Translator
translator = Translator()

medicines = load_medicines()
qa_pipeline = load_qa_model()
sentence_model = load_sentence_transformer_model()

st.title("Medicines Information System")

# Select language
language = st.radio("Select Language:", ('English', 'Burmese'))

# Text input for asking a question
question = st.text_input("Ask a question about any medicine:")

def build_relevant_context(medicine):
    context = ""
    context += f"Generic Name: {medicine['generic_name']}\n"
    context += f"Brand Names: {', '.join(medicine['brand_names'])}\n"
    context += f"Description: {medicine['description']}\n"
    context += f"Uses: {medicine['uses']}\n"
    context += f"Indications: {', '.join(medicine['indications'])}\n"
    context += f"Contraindications: {', '.join(medicine['contraindications'])}\n"
    context += "Side Effects: Common: " + ", ".join(medicine['side_effects']['common']) + "; Serious: " + ", ".join(medicine['side_effects']['serious']) + "\n"
    interactions = "; ".join([f"{i['drug']}: {i['description']}" for i in medicine['interactions']])
    context += f"Interactions: {interactions}\n"
    context += f"Warnings: {', '.join(medicine['warnings'])}\n"
    context += f"Mechanism of Action: {medicine['mechanism_of_action']}\n"
    pharmacokinetics = f"Pharmacokinetics: Absorption: {medicine['pharmacokinetics']['absorption']}; Metabolism: {medicine['pharmacokinetics']['metabolism']}; Half-life: {medicine['pharmacokinetics']['half_life']}; Excretion: {medicine['pharmacokinetics']['excretion']}"
    context += f"{pharmacokinetics}\n"
    context += f"Patient Information: {', '.join(medicine['patient_information'])}\n"
    return context

def get_specific_answer(question, medicine):
    if 'used for' in question.lower() or 'uses' in question.lower() or 'ဘာလဲ' in question.lower() or 'အသုံးပြုသည်' in question.lower():
        return medicine['uses']
    if 'side effects' in question.lower() or 'ဘေးထွက်ဆိုးကျိုး' in question.lower():
        return f"Common side effects include {', '.join(medicine['side_effects']['common'])}. Serious side effects include {', '.join(medicine['side_effects']['serious'])}."
    if 'brand names' in question.lower() or 'အမှတ်တံဆိပ်' in question.lower():
        return ", ".join(medicine['brand_names'])
    if 'contraindications' in question.lower() or 'ဆေးခံ့ကန့်ချက်' in question.lower():
        return ", ".join(medicine['contraindications'])
    if 'mechanism of action' in question.lower() or 'လုပ်ဆောင်ချက်' in question.lower():
        return medicine['mechanism_of_action']
    if 'how should i take' in question.lower() or 'ဘယ်လိုသောက်သင့်' in question.lower():
        return " ".join(medicine['patient_information'])
    return ""

def find_relevant_medicine(question, medicines):
    for drug in medicines:
        if drug['generic_name'].lower() in question.lower() or any(bn.lower() in question.lower() for bn in drug['brand_names']):
            return drug
    return None

def translate_text(text, src='en', dest='my'):
    try:
        return translator.translate(text, src=src, dest=dest).text
    except Exception as e:
        return text

def translate_if_needed(text, src, dest):
    if language == 'Burmese':
        return translate_text(text, src=src, dest=dest)
    return text

if question:
    # Translate question to English if in Burmese
    if language == 'Burmese':
        question = translate_text(question, src='my', dest='en')

    # Directly handle some common questions
    relevant_medicine = find_relevant_medicine(question, medicines)
    if relevant_medicine:
        specific_answer = get_specific_answer(question, relevant_medicine)
        if specific_answer:
            specific_answer_en = specific_answer
            specific_answer_my = translate_text(specific_answer, src='en', dest='my')
            st.write("Short Answer (English):", specific_answer_en)
            st.write("Short Answer (Burmese):", specific_answer_my)
        else:
            # Build the context relevant to the question using semantic search
            context = build_relevant_context(relevant_medicine)
            try:
                # Get the short and long answers
                short_answer = qa_pipeline(question=question, context=context)['answer']
                short_answer_en = short_answer
                short_answer_my = translate_text(short_answer, src='en', dest='my')
                st.write("Short Answer (English):", short_answer_en)
                st.write("Short Answer (Burmese):", short_answer_my)

                # Option to view detailed answer
                if st.button("Show Detailed Answer"):
                    detailed_answer_en = context
                    detailed_answer_my = translate_text(context, src='en', dest='my')
                    st.write("Detailed Answer (English):", detailed_answer_en)
                    st.write("Detailed Answer (Burmese):", detailed_answer_my)
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
    question = test["question"]
    expected_answer = test["expected"]
    
    # Translate question to English if in Burmese
    if language == 'Burmese':
        question = translate_text(question, src='my', dest='en')
    
    relevant_medicine = find_relevant_medicine(question, medicines)
    if relevant_medicine:
        specific_answer = get_specific_answer(question, relevant_medicine)
        if specific_answer:
            specific_answer_en = specific_answer
            specific_answer_my = translate_text(specific_answer, src='en', dest='my')
            st.write(f"**Question:** {test['question']}")
            st.write(f"**Expected Answer (English):** {expected_answer}")
            st.write(f"**Expected Answer (Burmese):** {translate_text(expected_answer, src='en', dest='my')}")
            st.write(f"**Model's Short Answer (English):** {specific_answer_en}")
            st.write(f"**Model's Short Answer (Burmese):** {specific_answer_my}")
            
            if st.button(f"Show Detailed Answer for '{test['question']}'"):
                context = build_relevant_context(relevant_medicine)
                detailed_answer_en = context
                detailed_answer_my = translate_text(context, src='en', dest='my')
                st.write(f"**Model's Detailed Answer (English):** {detailed_answer_en}")
                st.write(f"**Model's Detailed Answer (Burmese):** {detailed_answer_my}")
        else:
            context = build_relevant_context(relevant_medicine)
            try:
                short_answer = qa_pipeline(question=question, context=context)['answer']
                short_answer_en = short_answer
                short_answer_my = translate_text(short_answer, src='en', dest='my')
                st.write(f"**Question:** {test['question']}")
                st.write(f"**Expected Answer (English):** {expected_answer}")
                st.write(f"**Expected Answer (Burmese):** {translate_text(expected_answer, src='en', dest='my')}")
                st.write(f"**Model's Short Answer (English):** {short_answer_en}")
                st.write(f"**Model's Short Answer (Burmese):** {short_answer_my}")
                
                if st.button(f"Show Detailed Answer for '{test['question']}'"):
                    detailed_answer_en = context
                    detailed_answer_my = translate_text(context, src='en', dest='my')
                    st.write(f"**Model's Detailed Answer (English):** {detailed_answer_en}")
                    st.write(f"**Model's Detailed Answer (Burmese):** {detailed_answer_my}")
            except Exception as e:
                st.write(f"**Question:** {test['question']}")
                st.write(f"**Expected Answer (English):** {expected_answer}")
                st.write(f"**Expected Answer (Burmese):** {translate_text(expected_answer, src='en', dest='my')}")
                st.write(f"**Model's Short Answer:** An error occurred: {e}")
    else:
        st.write(f"**Question:** {test['question']}")
        st.write(f"**Expected Answer (English):** {expected_answer}")
        st.write(f"**Expected Answer (Burmese):** {translate_text(expected_answer, src='en', dest='my')}")
        st.write("**Model's Short Answer:** No relevant context found for the question.")
    st.write("---")
