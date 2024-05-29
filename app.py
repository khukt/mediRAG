import streamlit as st
import json
from transformers import pipeline

# Load JSON data
@st.cache
def load_data(file):
    try:
        with open(file) as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading {file}: {e}")
        return None

# Load all data
medicines = load_data('medicines.json')['medicines']
symptoms = load_data('symptoms.json')['symptoms']
indications = load_data('indications.json')['indications']
diseases = load_data('diseases.json')['diseases']
side_effects = load_data('side_effects.json')['side_effects']
interactions = load_data('interactions.json')['interactions']
warnings = load_data('warnings.json')['warnings']
contraindications = load_data('contraindications.json')['contraindications']
mechanisms_of_action = load_data('mechanisms_of_action.json')['mechanisms_of_action']
brand_names = load_data('brand_names.json')['brand_names']
generic_names = load_data('generic_names.json')['generic_names']
manufacturers = load_data('manufacturers.json')['manufacturers']
relationships = load_data('relationships.json')

# Convert to dictionaries for easy lookup
def to_dict(data, key='id'):
    return {item[key]: item for item in data}

medicines_dict = to_dict(medicines)
symptoms_dict = to_dict(symptoms)
indications_dict = to_dict(indications)
diseases_dict = to_dict(diseases)
side_effects_dict = to_dict(side_effects)
interactions_dict = to_dict(interactions)
warnings_dict = to_dict(warnings)
contraindications_dict = to_dict(contraindications)
mechanisms_dict = to_dict(mechanisms_of_action)
brand_names_dict = to_dict(brand_names)
generic_names_dict = to_dict(generic_names)
manufacturers_dict = to_dict(manufacturers)

# Load language model
@st.cache(allow_output_mutation=True)
def load_language_model():
    return pipeline("text-generation", model="distilgpt2")

language_model = load_language_model()

# Streamlit UI
st.set_page_config(page_title="Advanced Medicine Knowledge Base", page_icon="💊")

# Title and Description
st.title('💊 Advanced Medicine Knowledge Base')
st.write("""
Welcome to the Advanced Medicine Knowledge Base! Use the search functionality below to ask questions and get human-like answers based on our comprehensive database.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
search_type = st.sidebar.selectbox('Search by', ['Ask a Question', 'Medicine', 'Symptom', 'Indication', 'Disease'])

# Main Content Area
if search_type == 'Ask a Question':
    st.header('Ask a Question')
    st.write("Enter your question below and get a detailed answer based on the medicine database.")
    query = st.text_input('Enter your query')
    if query:
        context = " ".join([med['description'] for med in medicines])  # Simplified context
        with st.spinner('Generating response...'):
            response = language_model(f"Context: {context}\nQuestion: {query}\nAnswer:", max_length=150, num_return_sequences=1)
        st.success("Answer:")
        st.write(response[0]['generated_text'])

elif search_type == 'Medicine':
    st.header('Search by Medicine')
    medicine_options = {med['name']: med['id'] for med in medicines}
    selected_medicine = st.selectbox('Select a Medicine', list(medicine_options.keys()))
    
    if selected_medicine:
        medicine_id = medicine_options[selected_medicine]
        st.write(f"**Description:** {medicines_dict[medicine_id]['description']}")

        with st.expander("Related Symptoms"):
            related_symptoms = find_related_entities(medicine_id, 'medicine_symptom')
            display_related_entities(related_symptoms, symptoms_dict, 'symptom_id')

        with st.expander("Related Indications"):
            related_indications = find_related_entities(medicine_id, 'medicine_indication')
            display_related_entities(related_indications, indications_dict, 'indication_id')

        with st.expander("Related Diseases"):
            related_diseases = find_related_entities(medicine_id, 'medicine_disease')
            display_related_entities(related_diseases, diseases_dict, 'disease_id')

        with st.expander("Related Side Effects"):
            related_side_effects = find_related_entities(medicine_id, 'medicine_side_effect')
            display_related_entities(related_side_effects, side_effects_dict, 'side_effect_id')

        with st.expander("Related Interactions"):
            related_interactions = find_related_entities(medicine_id, 'medicine_interaction')
            display_related_entities(related_interactions, interactions_dict, 'interaction_id')

        with st.expander("Related Warnings"):
            related_warnings = find_related_entities(medicine_id, 'medicine_warning')
            display_related_entities(related_warnings, warnings_dict, 'warning_id')

        with st.expander("Related Contraindications"):
            related_contraindications = find_related_entities(medicine_id, 'medicine_contraindication')
            display_related_entities(related_contraindications, contraindications_dict, 'contraindication_id')

        with st.expander("Related Mechanisms of Action"):
            related_mechanisms = find_related_entities(medicine_id, 'medicine_mechanism')
            display_related_entities(related_mechanisms, mechanisms_dict, 'mechanism_id')

elif search_type == 'Symptom':
    st.header('Search by Symptom')
    symptom_options = {sym['name']: sym['id'] for sym in symptoms}
    selected_symptom = st.selectbox('Select a Symptom', list(symptom_options.keys()))

    if selected_symptom:
        symptom_id = symptom_options[selected_symptom]
        
        with st.expander("Related Medicines"):
            related_medicines = find_related_entities(symptom_id, 'medicine_symptom', 'symptom_id')
            display_related_entities(related_medicines, medicines_dict, 'medicine_id')

elif search_type == 'Indication':
    st.header('Search by Indication')
    indication_options = {ind['name']: ind['id'] for ind in indications}
    selected_indication = st.selectbox('Select an Indication', list(indication_options.keys()))

    if selected_indication:
        indication_id = indication_options[selected_indication]

        with st.expander("Related Medicines"):
            related_medicines = find_related_entities(indication_id, 'medicine_indication', 'indication_id')
            display_related_entities(related_medicines, medicines_dict, 'medicine_id')

elif search_type == 'Disease':
    st.header('Search by Disease')
    disease_options = {dis['name']: dis['id'] for dis in diseases}
    selected_disease = st.selectbox('Select a Disease', list(disease_options.keys()))

    if selected_disease:
        disease_id = disease_options[selected_disease]

        with st.expander("Related Medicines"):
            related_medicines = find_related_entities(disease_id, 'medicine_disease', 'disease_id')
            display_related_entities(related_medicines, medicines_dict, 'medicine_id')

# Footer
st.sidebar.markdown("""
---
*Developed by [Your Name](https://your-website.com)*
""")

# Run the app
if __name__ == '__main__':
    st.write("Welcome to the Medicine Knowledge Base")
