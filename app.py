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

def explain_answer_process(original_question, translated_question, relevant_medicine, specific_answer, context, short_answer):
    explanation = f"""
    **Explainable AI Process:**
    
    1. **Original Question:** {original_question}
    2. **Translated Question:** {translated_question}
    3. **Relevant Medicine Found:** {relevant_medicine['generic_name']}
    4. **Direct Answer Extraction:** {specific_answer}
    5. **Context Built:** {context[:500]}... (truncated for brevity)
    6. **Model's Short Answer:** {short_answer}
    """
    return explanation

if question:
    # Translate question to English if in Burmese
    if language == 'Burmese':
        original_question = question
        question = translate_text(question, src='my', dest='en')
    else:
        original_question = question

    # Directly handle some common questions
    relevant_medicine = find_relevant_medicine(question, medicines)
    if relevant_medicine:
        specific_answer = get_specific_answer(question, relevant_medicine)
        if specific_answer:
            specific_answer_en = specific_answer
            specific_answer_my = translate_text(specific_answer, src='en', dest='my')
            st.write("Short Answer (English):", specific_answer_en)
            st.write("Short Answer (Burmese):", specific_answer_my)
            
            # Explainable AI
            explanation = explain_answer_process(original_question, question, relevant_medicine, specific_answer, "", "")
            st.write(explanation)
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

                # Explainable AI
                explanation = explain_answer_process(original_question, question, relevant_medicine, "", context, short_answer)
                st.write(explanation)

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

# Explanation of the AI
st.subheader("About this AI System")
st.write("""
This AI system leverages advanced natural language processing (NLP) models to provide accurate and detailed information about medicines. The system uses the following technologies:

- **Multilingual XLM-RoBERTa Model:** This model is capable of understanding and processing questions in multiple languages, including English and Burmese.
- **Google Translator:** This tool is used to translate questions and answers between English and Burmese, ensuring that the system can respond in both languages.
- **Sentence Transformers:** These models are used for semantic search, allowing the system to find the most relevant information from the database based on the user's question.

### Responsible AI
- **Transparency:** The system provides clear and detailed answers, showing both the original and translated texts to ensure transparency.
- **Fairness:** The system is designed to provide accurate information regardless of the user's language, ensuring fair access to information.
- **Accountability:** The system includes mechanisms to handle errors and provide feedback, helping to improve the model's performance over time.
- **Privacy:** The system does not store any personal information from users, ensuring that user data remains private and secure.

If you have any questions or feedback about the system, please feel free to reach out.
""")
