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

# Streamlit app setup
st.set_page_config(page_title="Medicines Information System", layout="wide")

# Add custom CSS
st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #f0f2f6;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput input {
        background-color: #ffffff;
        color: #333333;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("💊 Medicines Information System")

# Select language
language = st.radio("Select Language:", ('English', 'Burmese'))

# Text input for asking a question
question = st.text_input("Ask a question about any medicine:")

def build_relevant_context(medicine):
    context = (
        f"Generic Name: {medicine['generic_name']}\n"
        f"Brand Names: {', '.join(medicine['brand_names'])}\n"
        f"Description: {medicine['description']}\n"
        f"Uses: {medicine['uses']}\n"
        f"Indications: {', '.join(medicine['indications'])}\n"
        f"Contraindications: {', '.join(medicine['contraindications'])}\n"
        f"Side Effects: Common: {', '.join(medicine['side_effects']['common'])}; Serious: {', '.join(medicine['side_effects']['serious'])}\n"
        f"Interactions: {'; '.join([f'{i['drug']}: {i['description']}' for i in medicine['interactions']])}\n"
        f"Warnings: {', '.join(medicine['warnings'])}\n"
        f"Mechanism of Action: {medicine['mechanism_of_action']}\n"
        f"Pharmacokinetics: Absorption: {medicine['pharmacokinetics']['absorption']}; Metabolism: {medicine['pharmacokinetics']['metabolism']}; Half-life: {medicine['pharmacokinetics']['half_life']}; Excretion: {medicine['pharmacokinetics']['excretion']}\n"
        f"Patient Information: {', '.join(medicine['patient_information'])}\n"
    )
    return context

def get_specific_answer(question, medicine):
    if any(keyword in question.lower() for keyword in ['used for', 'uses', 'ဘာအတွက်', 'အသုံးပြု']):
        return medicine['uses']
    if any(keyword in question.lower() for keyword in ['side effects', 'ဘေးထွက်ဆိုးကျိုး']):
        return f"Common side effects include {', '.join(medicine['side_effects']['common'])}. Serious side effects include {', '.join(medicine['side_effects']['serious'])}."
    if any(keyword in question.lower() for keyword in ['brand names', 'အမှတ်တံဆိပ်']):
        return ", ".join(medicine['brand_names'])
    if any(keyword in question.lower() for keyword in ['contraindications', 'ဆေးခံ့ကန့်ချက်']):
        return ", ".join(medicine['contraindications'])
    if any(keyword in question.lower() for keyword in ['mechanism of action', 'လုပ်ဆောင်ချက်']):
        return medicine['mechanism_of_action']
    if any(keyword in question.lower() for keyword in ['how should i take', 'ဘယ်လိုသောက်သင့်']):
        return " ".join(medicine['patient_information'])
    return ""

def find_relevant_medicine(question, medicines):
    question_embedding = sentence_model.encode(question, convert_to_tensor=True)
    highest_score = float('-inf')
    best_medicine = None

    for medicine in medicines:
        context = build_relevant_context(medicine)
        context_embedding = sentence_model.encode(context, convert_to_tensor=True)
        score = util.pytorch_cos_sim(question_embedding, context_embedding).item()

        if score > highest_score:
            highest_score = score
            best_medicine = medicine

    return best_medicine

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

def explain_detailed_process(original_question, translated_question, relevant_medicine, context, short_answer, detailed_answer):
    explanation = f"""
    **Explainable AI Process:**
    
    1. **Original Question:** {original_question}
    2. **Translated Question:** {translated_question}
    3. **Relevant Medicine Found:** {relevant_medicine['generic_name']}
    4. **Context Built:** {context[:500]}... (truncated for brevity)
    5. **Model's Short Answer:** {short_answer}
    6. **Detailed Answer:** {detailed_answer[:500]}... (truncated for brevity)
    """
    return explanation

if question:
    if language == 'Burmese':
        original_question = question
        question = translate_text(question, src='my', dest='en')
    else:
        original_question = question

    relevant_medicine = find_relevant_medicine(question, medicines)
    if relevant_medicine:
        specific_answer = get_specific_answer(question, relevant_medicine)
        if specific_answer:
            specific_answer_en = specific_answer
            specific_answer_my = translate_text(specific_answer, src='en', dest='my')
            st.markdown(f"### Short Answer (English): {specific_answer_en}")
            st.markdown(f"### Short Answer (Burmese): {specific_answer_my}")
            
            explanation = explain_answer_process(original_question, question, relevant_medicine, specific_answer, "", "")
            st.markdown(f"### Explainable AI Process:\n{explanation}")
        else:
            context = build_relevant_context(relevant_medicine)
            try:
                short_answer = qa_pipeline(question=question, context=context)['answer']
                short_answer_en = short_answer
                short_answer_my = translate_text(short_answer, src='en', dest='my')
                st.markdown(f"### Short Answer (English): {short_answer_en}")
                st.markdown(f"### Short Answer (Burmese): {short_answer_my}")

                explanation = explain_answer_process(original_question, question, relevant_medicine, "", context, short_answer)
                st.markdown(f"### Explainable AI Process:\n{explanation}")

                if st.button("Show Detailed Answer"):
                    detailed_answer_en = context
                    detailed_answer_my = translate_text(context, src='en', dest='my')
                    st.markdown(f"### Detailed Answer (English): {detailed_answer_en}")
                    st.markdown(f"### Detailed Answer (Burmese): {detailed_answer_my}")

                    detailed_explanation = explain_detailed_process(original_question, question, relevant_medicine, context, short_answer, context)
                    st.markdown(f"### Explainable AI Process:\n{detailed_explanation}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.markdown("### No relevant context found for the question.")

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
