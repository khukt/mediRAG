# Ensure the required packages are installed:
# !pip install shap transformers sentence-transformers googletrans==4.0.0-rc1 streamlit

import shap
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import json
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator
import matplotlib.pyplot as plt

# Load the medicines data from the JSON file
@st.cache_resource
def load_medicines():
    try:
        with open('medicines.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("The file 'medicines.json' was not found.")
        return []
    except json.JSONDecodeError:
        st.error("The file 'medicines.json' is not a valid JSON.")
        return []

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
    """Builds a detailed textual context for a given medicine."""
    context = ""
    context += f"**Generic Name:** {medicine.get('generic_name', 'N/A')}\n"
    context += f"**Brand Names:** {', '.join(medicine.get('brand_names', []))}\n"
    context += f"**Description:** {medicine.get('description', 'N/A')}\n"
    context += f"**Uses:** {medicine.get('uses', 'N/A')}\n"
    context += f"**Indications:** {', '.join(medicine.get('indications', []))}\n"
    context += f"**Contraindications:** {', '.join(medicine.get('contraindications', []))}\n"
    side_effects = medicine.get('side_effects', {})
    context += "**Side Effects:** Common: " + ", ".join(side_effects.get('common', [])) + "; Serious: " + ", ".join(side_effects.get('serious', [])) + "\n"
    interactions = "; ".join([f"{i['drug']}: {i['description']}" for i in medicine.get('interactions', [])])
    context += f"**Interactions:** {interactions}\n"
    context += f"**Warnings:** {', '.join(medicine.get('warnings', []))}\n"
    context += f"**Mechanism of Action:** {medicine.get('mechanism_of_action', 'N/A')}\n"
    pharmacokinetics = medicine.get('pharmacokinetics', {})
    context += f"**Pharmacokinetics:** Absorption: {pharmacokinetics.get('absorption', 'N/A')}; Metabolism: {pharmacokinetics.get('metabolism', 'N/A')}; Half-life: {pharmacokinetics.get('half_life', 'N/A')}; Excretion: {pharmacokinetics.get('excretion', 'N/A')}\n"
    context += f"**Patient Information:** {', '.join(medicine.get('patient_information', []))}\n"
    return context

def get_specific_answer(question, medicine):
    """Gets a specific answer from the medicine data based on the question."""
    if 'used for' in question.lower() or 'uses' in question.lower() or 'ဘာအတွက်' in question.lower() or 'အသုံးပြု' in question.lower():
        return medicine.get('uses', 'Information not available.')
    if 'side effects' in question.lower() or 'ဘေးထွက်ဆိုးကျိုး' in question.lower():
        side_effects = medicine.get('side_effects', {})
        return f"Common side effects include {', '.join(side_effects.get('common', []))}. Serious side effects include {', '.join(side_effects.get('serious', []))}."
    if 'brand names' in question.lower() or 'အမှတ်တံဆိပ်' in question.lower():
        return ", ".join(medicine.get('brand_names', []))
    if 'contraindications' in question.lower() or 'ဆေးခံ့ကန့်ချက်' in question.lower():
        return ", ".join(medicine.get('contraindications', []))
    if 'mechanism of action' in question.lower() or 'လုပ်ဆောင်ချက်' in question.lower():
        return medicine.get('mechanism_of_action', 'Information not available.')
    if 'how should i take' in question.lower() or 'ဘယ်လိုသောက်သင့်' in question.lower():
        return " ".join(medicine.get('patient_information', []))
    return ""

def find_relevant_medicine(question, medicines):
    """Finds the most relevant medicine based on the question using semantic similarity."""
    # Prepare question embedding
    question_embedding = sentence_model.encode(question, convert_to_tensor=True)

    best_medicine = None
    highest_score = float('-inf')

    # Iterate through each medicine to find the best match
    for medicine in medicines:
        context = build_relevant_context(medicine)
        context_embedding = sentence_model.encode(context, convert_to_tensor=True)
        score = util.pytorch_cos_sim(question_embedding, context_embedding).item()

        if score > highest_score:
            highest_score = score
            best_medicine = medicine

    return best_medicine

def translate_text(text, src='en', dest='my'):
    """Translates text from one language to another."""
    try:
        return translator.translate(text, src=src, dest=dest).text
    except Exception as e:
        return text

if question:
    # Translate question to English if in Burmese
    if language == 'Burmese':
        original_question = question
        question = translate_text(question, src='my', dest='en')
    else:
        original_question = question

    # Find the most relevant medicine using sentence transformers
    relevant_medicine = find_relevant_medicine(question, medicines)
    if relevant_medicine:
        specific_answer = get_specific_answer(question, relevant_medicine)
        if specific_answer:
            specific_answer_en = specific_answer
            specific_answer_my = translate_text(specific_answer, src='en', dest='my')
            st.write("### Short Answer (English):", specific_answer_en)
            st.write("### Short Answer (Burmese):", specific_answer_my)
        else:
            # Build the context relevant to the question using semantic search
            context = build_relevant_context(relevant_medicine)
            try:
                # Get the short answer
                short_answer = qa_pipeline(question=question, context=context)['answer']
                short_answer_en = short_answer
                short_answer_my = translate_text(short_answer, src='en', dest='my')
                st.write("### Short Answer (English):", short_answer_en)
                st.write("### Short Answer (Burmese):", short_answer_my)

                # SHAP Explanation
                explainer = shap.Explainer(qa_pipeline)
                shap_values = explainer({"question": question, "context": context})

                st.subheader("SHAP Explanation")
                fig, ax = plt.subplots(figsize=(10, 5))
                shap.waterfall_plot(shap_values[0], max_display=10)
                st.pyplot(fig)

                # Option to view detailed answer
                if st.button("Show Detailed Answer"):
                    detailed_answer_en = context
                    detailed_answer_my = translate_text(context, src='en', dest='my')
                    st.write("### Detailed Answer (English):", detailed_answer_en)
                    st.write("### Detailed Answer (Burmese):", detailed_answer_my)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.write("No relevant context found for the question.")
