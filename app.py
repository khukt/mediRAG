import streamlit as st
import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

# Set page configuration
st.set_page_config(page_title="Medicine Information Retrieval", layout="wide")

# Function to load JSON data with caching
@st.cache_data
def load_json_data(file_path):
    with open(file_path) as f:
        return json.load(f)

medicines = load_json_data('medicines.json')['medicines']
brand_names = load_json_data('brand_names.json')['brand_names']
manufacturers = load_json_data('manufacturers.json')['manufacturers']

# Create dictionaries for quick lookups
brand_dict = {brand['id']: brand for brand in brand_names}
manufacturer_dict = {manufacturer['id']: manufacturer for manufacturer in manufacturers}

# Load the multilingual NLP model with caching
@st.cache_resource
def load_nlp_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-multilingual-cased")
    qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=tokenizer)
    return tokenizer, qa_pipeline

tokenizer, qa_pipeline = load_nlp_model()

# Function to detect if a string contains Burmese characters
def contains_burmese(text):
    burmese_characters = set("ကခဂဃငစဆဇဈညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအဣဤဥဦဧဩဪါာိီုူေဲံ့းွှဿ၀၁၂၃၄၅၆၇၈၉")
    return any(char in burmese_characters for char in text)

# Function to parse the query and retrieve medicine information
def parse_query(query):
    query = query.lower()
    results = []

    for med in medicines:
        if query in med['generic_name'].lower() or query in med.get('generic_name_mm', '').lower():
            results.append(med)
    
    return results

# Function to generate context for question-answering
def generate_context(med_names):
    context_parts = []
    for med in medicines:
        if med['generic_name'].lower() in med_names or med.get('generic_name_mm', '').lower() in med_names:
            context_parts.append(f"{med['generic_name']} ({med.get('generic_name_mm', '')}): Uses: {', '.join(med['uses'])}. Side Effects: {', '.join(med['side_effects'])}.")
    return " ".join(context_parts)

# Function to answer complex questions
def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Sidebar with language switcher
language = st.sidebar.selectbox("Select Language", ["Burmese", "English"])

if language == "Burmese":
    st.sidebar.title("ညွှန်ကြားချက်များ")
    st.sidebar.write("""
    မေးမြန်းမှုအကြောင်းအရာကို (အထွေထွေသော နာမည် သို့မဟုတ် အမှတ်တံဆိပ်နာမည် သို့မဟုတ် ရောဂါလက္ခဏာ) အထဲသို့ ရိုက်ထည့်ပါ။
    မေးမြန်းမှုအရ ဆေးဝါးအကြောင်းအရာများကို ပြသပေးပါမည်။ 
    မြန်မာဘာသာဖြင့် ရှာဖွေနိုင်သည်။
    """)
    st.title('ဆေးဝါးအကြောင်း အချက်အလက် ရှာဖွေမှု')
    st.write('ဆေးဝါးအမည် (အထွေထွေ နာမည် သို့မဟုတ် အမှတ်တံဆိပ် နာမည်) သို့မဟုတ် ရောဂါလက္ခဏာအား ရိုက်ထည့်ပါ။')
    query_label = 'မေးမြန်းမှု'
else:
    st.sidebar.title("Instructions")
    st.sidebar.write("""
    Enter your query about a medicine name (generic or brand) or a symptom in the input box. 
    The app will display relevant medicine information based on your query. You can search in either English or Burmese.
    """)
    st.title('Medicine Information Retrieval')
    st.write('Enter your query about a medicine name (generic or brand) or a symptom.')
    query_label = 'Query'

# Display GENI logo, URL, and research team information
st.sidebar.image("https://www.geni.asia/wp-content/uploads/2020/09/geni-logo.png", use_column_width=True)
st.sidebar.write("[GENI Research Team](https://geni.asia)")
st.sidebar.write("This is part of CareDiary development by the GENI Research Team.")

# Add disclaimers
st.sidebar.write("**Disclaimer:** This is a demo transformer-based data retrieval system for educational purposes only. It is not intended for medical use. Please consult a healthcare professional for medical advice.")

query = st.text_input(query_label)

if query:
    # Check if the query is a complex question
    if "difference" in query.lower():
        # Extract the medicine names from the query
        med_names = set(word.strip() for word in query.lower().replace("difference between", "").replace("and", ",").split(","))
        # Generate context from the extracted medicine names
        context = generate_context(med_names)
        if context:
            answer = answer_question(query, context)
            st.write(f"**Answer:** {answer}")
        else:
            st.write("No relevant information found for the specified medicines.")
    else:
        results = parse_query(query)
        if results:
            for med in results:
                st.markdown(f"### {med['generic_name']} ({med.get('generic_name_mm', '')})")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("Uses (English)"):
                        st.write(', '.join(med['uses']))
                    with st.expander("Uses (Burmese)"):
                        if 'uses_mm' in med:
                            st.write(', '. join(med['uses_mm']))

                with col2:
                    with st.expander("Side Effects (English)"):
                        st.write(', '. join(med['side_effects']))
                    with st.expander("Side Effects (Burmese)"):
                        if 'side_effects_mm' in med:
                            st.write(', '. join(med['side_effects_mm']))

                st.markdown("**Brands and Dosages**")
                for brand_id in med['brand_names']:
                    brand = brand_dict[brand_id]
                    manufacturer = manufacturer_dict[brand['manufacturer_id']]
                    st.markdown(f"**Brand Name:** {brand['name']}")
                    st.write(f"**Dosages:** {', '. join(brand['dosages'])}")
                    st.write(f"**Manufacturer:** {manufacturer['name']}")
                    st.write(f"**Contact Info:**")
                    st.write(f"Phone: {manufacturer['contact_info']['phone']}")
                    st.write(f"Email: {manufacturer['contact_info']['email']}")
                    st.write(f"Address: {manufacturer['contact_info']['address']}")
                    st.markdown("---")
        else:
            st.write('No results found.')
