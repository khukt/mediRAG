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
generic_names = load_json_data('generic_names.json')['generic_names']
brand_names = load_json_data('brand_names.json')['brand_names']
manufacturers = load_json_data('manufacturers.json')['manufacturers']
forms = load_json_data('forms.json')['forms']
symptoms = load_json_data('symptoms.json')['symptoms']
diseases = load_json_data('diseases.json')['diseases']

# Create dictionaries for quick lookups
generic_dict = {generic['id']: generic for generic in generic_names}
brand_dict = {brand['id']: brand for brand in brand_names}
manufacturer_dict = {manufacturer['id']: manufacturer for manufacturer in manufacturers}
form_dict = {form['id']: form for form in forms}
symptom_dict = {symptom['id']: symptom for symptom in symptoms}
disease_dict = {disease['id']: disease for disease in diseases}

# Load the multilingual model with caching
@st.cache_resource
def load_nlp_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-multilingual-cased")
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return tokenizer, model, qa_pipeline

tokenizer, model, qa_pipeline = load_nlp_model()

# Function to detect if a string contains Burmese characters
def contains_burmese(text):
    burmese_characters = set("ကခဂဃငစဆဇဈညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအဣဤဥဦဧဩဪါာိီုူေဲံ့းွှဿ၀၁၂၃၄၅၆၇၈၉")
    return any(char in burmese_characters for char in text)

# Function to parse the query and retrieve medicine information
def parse_query(query):
    query = query.lower()
    results = []

    # Tokenize the query using the tokenizer
    inputs = tokenizer(query, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

    for med in medicines:
        generic_name_matches = any(token in generic_dict[gname_id]['name'].lower() for gname_id in med['generic_name_ids'] for token in tokens)
        generic_name_mm_matches = any(token in generic_dict[gname_id]['name_mm'].lower() for gname_id in med['generic_name_ids'] for token in tokens)
        indications_matches = any(token in ' '.join(med.get('indications', [])).lower() for token in tokens)
        indications_mm_matches = any(token in ' '.join(med.get('indications_mm', [])).lower() for token in tokens)
        side_effects_matches = any(token in ' '.join(med.get('side_effects', [])).lower() for token in tokens)
        side_effects_mm_matches = any(token in ' '.join(med.get('side_effects_mm', [])).lower() for token in tokens)
        contraindications_matches = any(token in ' '.join(med.get('contraindications', [])).lower() for token in tokens)
        contraindications_mm_matches = any(token in ' '.join(med.get('contraindications_mm', [])).lower() for token in tokens)
        warnings_matches = any(token in ' '.join(med.get('warnings', [])).lower() for token in tokens)
        warnings_mm_matches = any(token in ' '.join(med.get('warnings_mm', [])).lower() for token in tokens)
        interactions_matches = any(token in ' '.join(med.get('interactions', [])).lower() for token in tokens)
        interactions_mm_matches = any(token in ' '.join(med.get('interactions_mm', [])).lower() for token in tokens)
        brand_name_matches = any(any(token in brand_dict[brand_info['brand_id']]['name'].lower() for token in tokens) for brand_info in med['brands'])

        if (generic_name_matches or generic_name_mm_matches or indications_matches or indications_mm_matches or
            side_effects_matches or side_effects_mm_matches or contraindications_matches or contraindications_mm_matches or
            warnings_matches or warnings_mm_matches or interactions_matches or interactions_mm_matches or brand_name_matches):
            results.append(med)
    
    return results

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
        st.markdown("## Results")
        for med in results:
            st.markdown(f"### {generic_dict[med['generic_name_ids'][0]]['name']} ({generic_dict[med['generic_name_ids'][0]]['name_mm']})")
            st.write(f"**Description:** {med.get('description', 'N/A')} ({med.get('description_mm', 'N/A')})")
            st.write(f"**Mechanism of Action:** {med.get('mechanism_of_action', 'N/A')} ({med.get('mechanism_of_action_mm', 'N/A')})")
            
            col1, col2 = st.columns(2)
            
            with col1:
                with st.expander("Indications (English)"):
                    st.write(', '.join(med.get('indications', [])))
                with st.expander("Indications (Burmese)"):
                    st.write(', '.join(med.get('indications_mm', [])))

                with st.expander("Side Effects (English)"):
                    st.write(', '.join(med.get('side_effects', [])))
                with st.expander("Side Effects (Burmese)"):
                    st.write(', '.join(med.get('side_effects_mm', [])))

            with col2:
                with st.expander("Contraindications (English)"):
                    st.write(', '.join(med.get('contraindications', [])))
                with st.expander("Contraindications (Burmese)"):
                    st.write(', '.join(med.get('contraindications_mm', [])))

                with st.expander("Warnings (English)"):
                    st.write(', '.join(med.get('warnings', [])))
                with st.expander("Warnings (Burmese)"):
                    st.write(', '.join(med.get('warnings_mm', [])))

                with st.expander("Drug Interactions (English)"):
                    st.write(', '.join(med.get('interactions', [])))
                with st.expander("Drug Interactions (Burmese)"):
                    st.write(', '.join(med.get('interactions_mm', [])))

            st.markdown("**Brands and Dosages**")
            for brand_info in med['brands']:
                brand = brand_dict[brand_info['brand_id']]
                manufacturer = manufacturer_dict[brand['manufacturer_id']]
                form = form_dict[brand_info['form_id']]
                st.markdown(f"**Brand Name:** {brand['name']} ({brand['name_mm']})")
                st.write(f"**Dosages:** {', '.join(brand_info['dosages'])}")
                st.write(f"**Form:** {form['name']} ({form['name_mm']})")
                st.write(f"**Manufacturer:** {manufacturer['name']}")
                st.write(f"**Contact Info:**")
                st.write(f"Phone: {manufacturer['contact_info']['phone']}")
                st.write(f"Email: {manufacturer['contact_info']['email']}")
                st.write(f"Address: {manufacturer['contact_info']['address']}")
                st.markdown("---")

            # Display symptoms
            st.markdown("**Symptoms Treated**")
            for symptom_id in med['symptom_ids']:
                symptom = symptom_dict[symptom_id]
                st.write(f"{symptom['name']} ({symptom['name_mm']})")

            # Display diseases
            st.markdown("**Diseases Treated**")
            for disease_id in med['disease_ids']:
                disease = disease_dict[disease_id]
                st.write(f"{disease['name']} ({disease['name_mm']})")

            st.markdown("**Additional Information:**")
            st.write(f"{med.get('additional_info', 'N/A')} ({med.get('additional_info_mm', 'N/A')})")

        # Use the Q&A model to generate more user-friendly answers
        st.markdown("## Generated Answers")
        for med in results:
            context = f"Generic Name: {generic_dict[med['generic_name_ids'][0]]['name']}\n" \
                      f"Description: {med.get('description', 'N/A')}\n" \
                      f"Mechanism of Action: {med.get('mechanism_of_action', 'N/A')}\n" \
                      f"Indications: {', '.join(med.get('indications', []))}\n" \
                      f"Side Effects: {', '.join(med.get('side_effects', []))}\n" \
                      f"Contraindications: {', '.join(med.get('contraindications', []))}\n" \
                      f"Warnings: {', '.join(med.get('warnings', []))}\n" \
                      f"Drug Interactions: {', '.join(med.get('interactions', []))}\n" \
                      f"Brands: {', '.join(brand_dict[brand_info['brand_id']]['name'] for brand_info in med['brands'])}\n"
            qa_result = qa_pipeline(question=query, context=context)
            st.write(f"**Answer:** {qa_result['answer']}")
    else:
        st.write('No results found.')
