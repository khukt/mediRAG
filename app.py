import streamlit as st
import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Set page configuration
st.set_page_config(page_title="Medicine Information Retrieval", layout="wide")

# Data loading functions
@st.cache_data
def load_json_data(file_path):
    with open(file_path) as f:
        return json.load(f)

def load_all_data():
    medicines = load_json_data('medicines.json')['medicines']
    generic_names = load_json_data('generic_names.json')['generic_names']
    brand_names = load_json_data('brand_names.json')['brand_names']
    manufacturers = load_json_data('manufacturers.json')['manufacturers']
    forms = load_json_data('forms.json')['forms']
    symptoms = load_json_data('symptoms.json')['symptoms']
    diseases = load_json_data('diseases.json')['diseases']
    return medicines, generic_names, brand_names, manufacturers, forms, symptoms, diseases

medicines, generic_names, brand_names, manufacturers, forms, symptoms, diseases = load_all_data()

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

# Function to parse the query and prioritize results
def parse_query(query):
    query = query.lower()
    results = {
        "brand_names": [],
        "diseases": [],
        "generic_names": [],
        "manufacturers": [],
        "medicines": [],
        "symptoms": []
    }
    tokens = query.split()

    for token in tokens:
        for brand in brand_names:
            if token in brand['name'].lower() or token in brand['name_mm'].lower():
                results['brand_names'].append(brand)

        for disease in diseases:
            if token in disease['name'].lower() or token in disease['name_mm'].lower():
                results['diseases'].append(disease)

        for generic in generic_names:
            if token in generic['name'].lower() or token in generic['name_mm'].lower():
                results['generic_names'].append(generic)

        for manufacturer in manufacturers:
            if token in manufacturer['name'].lower():
                results['manufacturers'].append(manufacturer)

        for med in medicines:
            if any([
                any(token in generic_dict[gname_id]['name'].lower() for gname_id in med['generic_name_ids']),
                any(token in generic_dict[gname_id]['name_mm'].lower() for gname_id in med['generic_name_ids']),
                token in ' '.join(med.get('indications', [])).lower(),
                token in ' '.join(med.get('indications_mm', [])).lower(),
                token in ' '.join(med.get('side_effects', [])).lower(),
                token in ' '.join(med.get('side_effects_mm', [])).lower(),
                token in ' '.join(med.get('contraindications', [])).lower(),
                token in ' '.join(med.get('contraindications_mm', [])).lower(),
                token in ' '.join(med.get('warnings', [])).lower(),
                token in ' '.join(med.get('warnings_mm', [])).lower(),
                token in ' '.join(med.get('interactions', [])).lower(),
                token in ' '.join(med.get('interactions_mm', [])).lower()
            ]):
                results['medicines'].append(med)

        for symptom in symptoms:
            if token in symptom['name'].lower() or token in symptom['name_mm'].lower():
                results['symptoms'].append(symptom)

    return results

# Display functions
def display_brand_info(brand):
    manufacturer = manufacturer_dict[brand['manufacturer_id']]
    st.markdown(f"### {brand['name']} ({brand['name_mm']})")
    st.markdown(f"**Manufacturer:** {manufacturer['name']}")
    st.markdown(f"**Contact Info:**")
    st.write(f"Phone: {manufacturer['contact_info']['phone']}")
    st.write(f"Email: {manufacturer['contact_info']['email']}")
    st.write(f"Address: {manufacturer['contact_info']['address']}")
    st.markdown("---")

def display_disease_info(disease):
    st.markdown(f"### {disease['name']} ({disease['name_mm']})")
    st.markdown("---")

def display_generic_name_info(generic):
    st.markdown(f"### {generic['name']} ({generic['name_mm']})")
    st.markdown("---")

def display_manufacturer_info(manufacturer):
    st.markdown(f"### {manufacturer['name']}")
    st.markdown(f"**Contact Info:**")
    st.write(f"Phone: {manufacturer['contact_info']['phone']}")
    st.write(f"Email: {manufacturer['contact_info']['email']}")
    st.write(f"Address: {manufacturer['contact_info']['address']}")
    st.markdown("---")

def display_medicine_info(med, query_tokens):
    st.markdown(f"### {generic_dict[med['generic_name_ids'][0]]['name']} ({generic_dict[med['generic_name_ids'][0]]['name_mm']})")
    st.markdown(f"**Description:** {med.get('description', 'N/A')} ({med.get('description_mm', 'N/A')})")
    st.markdown(f"**Mechanism of Action:** {med.get('mechanism_of_action', 'N/A')} ({med.get('mechanism_of_action_mm', 'N/A')})")

    def highlight_matches(text, tokens):
        for token in tokens:
            text = text.replace(token, f"<mark>{token}</mark>")
        return text

    with st.expander("Indications"):
        st.markdown(f"**English:** {highlight_matches(', '.join(med.get('indications', [])), query_tokens)}", unsafe_allow_html=True)
        st.markdown(f"**Burmese:** {highlight_matches(', '.join(med.get('indications_mm', [])), query_tokens)}", unsafe_allow_html=True)

    with st.expander("Side Effects"):
        st.markdown(f"**English:** {highlight_matches(', '.join(med.get('side_effects', [])), query_tokens)}", unsafe_allow_html=True)
        st.markdown(f"**Burmese:** {highlight_matches(', '.join(med.get('side_effects_mm', [])), query_tokens)}", unsafe_allow_html=True)

    with st.expander("Contraindications"):
        st.markdown(f"**English:** {highlight_matches(', '.join(med.get('contraindications', [])), query_tokens)}", unsafe_allow_html=True)
        st.markdown(f"**Burmese:** {highlight_matches(', '.join(med.get('contraindications_mm', [])), query_tokens)}", unsafe_allow_html=True)

    with st.expander("Warnings"):
        st.markdown(f"**English:** {highlight_matches(', '.join(med.get('warnings', [])), query_tokens)}", unsafe_allow_html=True)
        st.markdown(f"**Burmese:** {highlight_matches(', '.join(med.get('warnings_mm', [])), query_tokens)}", unsafe_allow_html=True)

    with st.expander("Drug Interactions"):
        st.markdown(f"**English:** {highlight_matches(', '.join(med.get('interactions', [])), query_tokens)}", unsafe_allow_html=True)
        st.markdown(f"**Burmese:** {highlight_matches(', '.join(med.get('interactions_mm', [])), query_tokens)}", unsafe_allow_html=True)

    st.markdown("### Brands and Dosages")
    for brand_info in med['brands']:
        display_brand_info(brand_dict[brand_info['brand_id']])

    st.markdown("### Symptoms Treated")
    symptom_names = [f"{symptom_dict[symptom_id]['name']} ({symptom_dict[symptom_id]['name_mm']})" for symptom_id in med['symptom_ids']]
    st.write(', '.join(symptom_names))

    st.markdown("### Diseases Treated")
    disease_names = [f"{disease_dict[disease_id]['name']} ({disease_dict[disease_id]['name_mm']})" for disease_id in med['disease_ids']]
    st.write(', '.join(disease_names))

    st.markdown("### Additional Information")
    st.write(f"{med.get('additional_info', 'N/A')} ({med.get('additional_info_mm', 'N/A')})")

def display_symptom_info(symptom):
    st.markdown(f"### {symptom['name']} ({symptom['name_mm']})")
    st.markdown("---")

def display_generated_answers(query, med):
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
    st.markdown(f"**Answer:** {qa_result['answer']}")

# Sidebar and main layout
def main():
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

    st.sidebar.image("https://www.geni.asia/wp-content/uploads/2020/09/geni-logo.png", use_column_width=True)
    st.sidebar.write("[GENI Research Team](https://geni.asia)")
    st.sidebar.write("This is part of CareDiary development by the GENI Research Team.")
    st.sidebar.write("**Disclaimer:** This is a demo transformer-based data retrieval system for educational purposes only. It is not intended for medical use. Please consult a healthcare professional for medical advice.")

    query = st.text_input(query_label)

    if query:
        results = parse_query(query)
        query_tokens = query.split()
    
        if any(results.values()):
            st.markdown("### Search Results")

            if results['brand_names']:
                st.markdown("## Brand Names")
                for brand in results['brand_names']:
                    display_brand_info(brand)
                    # Additional info for brand
                    st.markdown("**Related Generic Names:**")
                    related_generics = [generic_dict[med['generic_name_ids'][0]]['name'] for med in medicines if any(b['brand_id'] == brand['id'] for b in med['brands'])]
                    st.write(', '.join(related_generics))
                    st.markdown("**Available Forms:**")
                    available_forms = [form_dict[b['form_id']]['name'] for med in medicines for b in med['brands'] if b['brand_id'] == brand['id']]
                    st.write(', '.join(available_forms))
        
            if results['diseases']:
                st.markdown("## Diseases")
                for disease in results['diseases']:
                    display_disease_info(disease)
                    # Additional info for disease
                    st.markdown("**Related Symptoms:**")
                    related_symptoms = [symptom_dict[symptom_id]['name'] for symptom_id in disease['symptom_ids']]
                    st.write(', '.join(related_symptoms))
                    st.markdown("**Recommended Medicines:**")
                    recommended_meds = [med['description'] for med in medicines if disease['id'] in med['disease_ids']]
                    st.write(', '.join(recommended_meds))

            if results['generic_names']:
                st.markdown("## Generic Names")
                for generic in results['generic_names']:
                    display_generic_name_info(generic)
                    # Additional info for generic name
                    st.markdown("**Associated Brand Names:**")
                    associated_brands = [brand['name'] for brand in brand_names if any(generic['id'] in med['generic_name_ids'] for med in medicines if any(b['brand_id'] == brand['id'] for b in med['brands']))]
                    st.write(', '.join(associated_brands))
                    st.markdown("**Indications:**")
                    indications = [med['description'] for med in medicines if generic['id'] in med['generic_name_ids']]
                    st.write(', '.join(indications))

            if results['manufacturers']:
                st.markdown("## Manufacturers")
                for manufacturer in results['manufacturers']:
                    display_manufacturer_info(manufacturer)
                    # Additional info for manufacturer
                    st.markdown("**Produced Medicines:**")
                    produced_meds = [brand['name'] for brand in brand_names if brand['manufacturer_id'] == manufacturer['id']]
                    st.write(', '.join(produced_meds))

            if results['medicines']:
                st.markdown("## Medicines")
                for med in results['medicines']:
                    display_medicine_info(med, query_tokens)
                    # Cross-reference to similar medicines
                    similar_meds = [m['description'] for m in medicines if any(ind in m['indications'] for ind in med['indications']) and m['id'] != med['id']]
                    if similar_meds:
                        st.markdown("**Similar Medicines:**")
                        st.write(', '.join(similar_meds))

            if results['symptoms']:
                st.markdown("## Symptoms")
                for symptom in results['symptoms']:
                    display_symptom_info(symptom)
                    # Additional info for symptom
                    st.markdown("**Potential Diseases:**")
                    potential_diseases = [disease_dict[disease_id]['name'] for disease_id in [d['id'] for d in diseases if symptom['id'] in d['symptom_ids']]]
                    st.write(', '.join(potential_diseases))
                    st.markdown("**Recommended Medicines:**")
                    recommended_meds = [med['description'] for med in medicines if symptom['id'] in med['symptom_ids']]
                    st.write(', '.join(recommended_meds))
        
            st.markdown("## Generated Answers")
            for med in results['medicines']:
                display_generated_answers(query, med)
    
        else:
            st.write('No results found.')

    if __name__ == "__main__":
        main()

