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

def build_relevant_context(question, medicines):
    context = ""
    keywords = question.lower().split()
    for drug in medicines:
        if any(kw in drug['generic_name'].lower() for kw in keywords) or any(kw.lower() in [bn.lower() for bn in drug['brand_names']] for kw in keywords):
            context += f"Generic Name: {drug['generic_name']}\n"
            context += f"Brand Names: {', '.join(drug['brand_names'])}\n"
            context += f"Description: {drug['description']}\n"
            context += f"Uses: {drug['uses']}\n"
            context += f"Indications: {', '.join(drug['indications'])}\n"
            context += f"Contraindications: {', '.join(drug['contraindications'])}\n"
            context += "Side Effects: Common: " + ", ".join(drug['side_effects']['common']) + "; Serious: " + ", ".join(drug['side_effects']['serious']) + "\n"
            interactions = "; ".join([f"{i['drug']}: {i['description']}" for i in drug['interactions']])
            context += f"Interactions: {interactions}\n"
            context += f"Warnings: {', '.join(drug['warnings'])}\n"
            context += f"Mechanism of Action: {drug['mechanism_of_action']}\n"
            pharmacokinetics = f"Pharmacokinetics: Absorption: {drug['pharmacokinetics']['absorption']}; Metabolism: {drug['pharmacokinetics']['metabolism']}; Half-life: {drug['pharmacokinetics']['half_life']}; Excretion: {drug['pharmacokinetics']['excretion']}"
            context += f"{pharmacokinetics}\n"
            context += f"Patient Information: {', '.join(drug['patient_information'])}\n"
    return context

def semantic_search(question, medicines, model):
    # Translate the question to English for semantic search if necessary
    if language == 'Burmese':
        question = translator.translate(question, src='my', dest='en').text
    
    # Embed the question using the sentence transformer model
    question_embedding = model.encode(question, convert_to_tensor=True)
    
    # Embed each medicine entry and compute similarity scores
    contexts = []
    for drug in medicines:
        context = build_relevant_context(drug['generic_name'], [drug])
        context_embedding = model.encode(context, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(question_embedding, context_embedding).item()
        contexts.append((context, similarity))
    
    # Sort contexts by similarity
    contexts = sorted(contexts, key=lambda x: x[1], reverse=True)
    return [context[0] for context in contexts[:3]] if contexts else []

def generate_answers(question, contexts, language):
    if language == 'Burmese':
        question_trans
