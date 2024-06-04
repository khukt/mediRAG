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
        question_translated = translator.translate(question, src='my', dest='en').text
    else:
        question_translated = question

    answers = []
    for context in contexts:
        short_answer = qa_pipeline(question=question_translated, context=context)
        start = short_answer['start']
        end = short_answer['end']
        full_sentence = context[max(0, start-50):min(len(context), end+50)]
        if question.lower() in full_sentence.lower():
            short_answer_text = full_sentence
        else:
            short_answer_text = short_answer['answer']
        
        if language == 'Burmese':
            short_answer_text = translator.translate(short_answer_text, src='en', dest='my').text
            context = translator.translate(context, src='en', dest='my').text
        
        answers.append(short_answer_text)
    
    return answers

def generate_direct_answer(question, medicines):
    keywords = question.lower().split()
    for drug in medicines:
        if 'paracetamol' in keywords or any(kw in drug['generic_name'].lower() for kw in keywords):
            if 'used for' in question.lower() or 'uses' in question.lower():
                return drug['uses']
            if 'side effects' in question.lower():
                return f"Common: {', '.join(drug['side_effects']['common'])}. Serious: {', '.join(drug['side_effects']['serious'])}."
            if 'brand names' in question.lower():
                return ", ".join(drug['brand_names'])
            if 'contraindications' in question.lower():
                return ", ".join(drug['contraindications'])
            if 'mechanism of action' in question.lower():
                return drug['mechanism_of_action']
            if 'how should i take' in question.lower():
                return " ".join(drug['patient_information'])
    return ""

if question:
    # Directly handle some common questions
    direct_answer = generate_direct_answer(question, medicines)
    if direct_answer:
        st.write("Short Answer:", direct_answer)
    else:
        # Build the context relevant to the question using semantic search
        contexts = semantic_search(question, medicines, sentence_model)
        if contexts:
            try:
                # Get the short and long answers
                answers = generate_answers(question, contexts, language)
                st.write("Short Answer:", answers[0])

                # Option to view detailed answer
                if st.button("Show Detailed Answer"):
                    detailed_answers = "\n\n".join(context for context in contexts)
                    st.write("Detailed Answer:", detailed_answers)
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
    direct_answer = generate_direct_answer(test["question"], medicines)
    if direct_answer:
        st.write(f"**Question:** {test['question']}")
        st.write(f"**Expected Answer:** {test['expected']}")
        st.write(f"**Model's Short Answer:** {direct_answer}")
        
        if st.button(f"Show Detailed Answer for '{test['question']}'"):
            contexts = semantic_search(test["question"], medicines, sentence_model)
            detailed_answers = "\n\n".join(context for context in contexts)
            st.write(f"**Model's Detailed Answer:** {detailed_answers}")
    else:
        contexts = semantic_search(test["question"], medicines, sentence_model)
        if contexts:
            try:
                answers = generate_answers(test["question"], contexts, language)
                st.write(f"**Question:** {test['question']}")
                st.write(f"**Expected Answer:** {test['expected']}")
                st.write(f"**Model's Short Answer:** {answers[0]}")
                
                if st.button(f"Show Detailed Answer for '{test['question']}'"):
                    detailed_answers = "\n\n".join(context for context in contexts)
                    st.write(f"**Model's Detailed Answer:** {detailed_answers}")
            except Exception as e:
                st.write(f"**Question:** {test['question']}")
                st.write(f"**Expected Answer:** {test['expected']}")
                st.write(f"**Model's Short Answer:** An error occurred: {e}")
        else:
            st.write(f"**Question:** {test['question']}")
            st.write(f"**Expected Answer:** {test['expected']}")
            st.write("**Model's Short Answer:** No relevant context found for the question.")
    st.write("---")
