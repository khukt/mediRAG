import streamlit as st
import json
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import torch

# Set page configuration
st.set_page_config(page_title="Medicine Information Retrieval", layout="wide")

# Function to load JSON data with caching
@st.cache_data
def load_json_data(file_path):
    with open(file_path) as f:
        return json.load(f)

# Load data
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
    model = AutoModelForMaskedLM.from_pretrained("distilbert-base-multilingual-cased")
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return tokenizer, model, qa_pipeline

tokenizer, model, qa_pipeline = load_nlp_model()

# Function to detect if a string contains Burmese characters
def contains_burmese(text):
    burmese_characters = set("ကခဂဃငစဆဇဈညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအဣဤဥဦဧဩဪါာိီုူေဲံ့းွှဿ၀၁၂၃၄၅၆၇၈၉")
    return any(char in burmese_characters for char in text)

# Function to classify the query type
def classify_query(query):
    if "difference between" in query or "differ" in query:
        return "comparison"
    elif "what is" in query or "ဆိုတာဘာလဲ" in query:
        return "simple"
    elif "tell me about" in query or "အကြောင်းပြောပြပါ" in query:
        return "brand"
    elif "for a headache" in query or "ခေါင်းကိုက်ကို" in query:
        return "symptom"
    return "unknown"

# Function to parse the query and retrieve medicine information
def parse_query(query):
    query_type = classify_query(query.lower())
    query = query.lower()

    if query_type == "simple":
        if contains_burmese(query):
            term = query.replace("ဆိုတာဘာလဲ", "").strip()
        else:
            term = query.replace("what is", "").strip()
        
        for med in medicines:
            if term in med['generic_name'].lower() or term in med.get('generic_name_mm', '').lower():
                return med

    elif query_type == "brand":
        if contains_burmese(query):
            term = query.replace("အကြောင်းပြောပြပါ", "").strip()
        else:
            term = query.replace("tell me about", "").strip()
        
        for brand in brand_names:
            if term in brand['name'].lower():
                med = next((m for m in medicines if brand['id'] in m['brand_names']), None)
                return med

    elif query_type == "symptom":
        if contains_burmese(query):
            term = "ခေါင်းကိုက်"
        else:
            term = "headache"
        
        results = []
        for med in medicines:
            if any(term in use.lower() for use in med['uses']) or any(term in use.lower() for use in med.get('uses_mm', [])):
                results.append(med)
        return results

    elif query_type == "comparison":
        meds = re.findall(r'between (.*?) and (.*)', query)
        if not meds:
            meds = re.findall(r'differ from (.*?) and (.*)', query)
        if meds:
            med1_name, med2_name = meds[0]
            med1 = next((m for m in medicines if med1_name.strip() in m['generic_name'].lower()), None)
            med2 = next((m for m in medicines if med2_name.strip() in m['generic_name'].lower()), None)
            return med1, med2

    return None

# Function to display medicine information
def display_medicine(med):
    st.markdown(f"### {med['generic_name']} ({med.get('generic_name_mm', '')})")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("Uses (English)"):
            st.write(', '.join(med['uses']))
        with st.expander("Uses (Burmese)"):
            if 'uses_mm' in med:
                st.write(', '.join(med['uses_mm']))

    with col2:
        with st.expander("Side Effects (English)"):
            st.write(', '.join(med['side_effects']))
        with st.expander("Side Effects (Burmese)"):
            if 'side_effects_mm' in med:
                st.write(', '.join(med['side_effects_mm']))

    st.markdown("**Brands and Dosages**")
    for brand_id in med['brand_names']:
        brand = brand_dict[brand_id]
        manufacturer = manufacturer_dict[brand['manufacturer_id']]
        st.markdown(f"**Brand Name:** {brand['name']}")
        st.write(f"**Dosages:** {', '.join(brand['dosages'])}")
        st.write(f"**Manufacturer:** {manufacturer['name']}")
        st.write(f"**Contact Info:**")
        st.write(f"Phone: {manufacturer['contact_info']['phone']}")
        st.write(f"Email: {manufacturer['contact_info']['email']}")
        st.write(f"Address: {manufacturer['contact_info']['address']}")
        st.markdown("---")

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
    results = parse_query(query)
    if results:
        if isinstance(results, list):
            for med in results:
                display_medicine(med)
        elif isinstance(results, tuple):
            med1, med2 = results
            st.markdown(f"### Comparison of {med1['generic_name']} and {med2['generic_name']}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### {med1['generic_name']}")
                display_medicine(med1)
            with col2:
                st.markdown(f"#### {med2['generic_name']}")
                display_medicine(med2)
        else:
            display_medicine(results)
    else:
        st.write('No results found.')
