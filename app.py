import streamlit as st
import json
from transformers import pipeline

# Load JSON data with debug information
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

# Function to find relationships
def find_related_entities(entity_id, relationship, entity_key='medicine_id'):
    related_ids = [rel for rel in relationships[relationship] if rel[entity_key] == entity_id]
    return related_ids

# Function to display related entities
def display_related_entities(related_ids, entity_dict, entity_key):
    for rel in related_ids:
        st.write(entity_dict[rel[entity_key]]['name'])

# Load RAG model
@st.cache(allow_output_mutation=True)
def load_rag_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

rag_model = load_rag_model()

# Streamlit UI
st.title('Medicine Knowledge Base')

# Search Type Selection
search_type = st.selectbox('Search by', ['Medicine', 'Symptom', 'Indication', 'Disease'])

if search_type == 'Medicine':
    # Medicine Selection
    medicine_options = {med['name']: med['id'] for med in medicines}
    selected_medicine = st.selectbox('Select a Medicine', list(medicine_options.keys()))
    
    if selected_medicine:
        medicine_id = medicine_options[selected_medicine]
        st.write(f"**Description:** {medicines_dict[medicine_id]['description']}")

        # Display related symptoms
        st.subheader('Related Symptoms')
        related_symptoms = find_related_entities(medicine_id, 'medicine_symptom')
        display_related_entities(related_symptoms, symptoms_dict, 'symptom_id')

        # Display related indications
        st.subheader('Related Indications')
        related_indications = find_related_entities(medicine_id, 'medicine_indication')
        display_related_entities(related_indications, indications_dict, 'indication_id')

        # Display related diseases
        st.subheader('Related Diseases')
        related_diseases = find_related_entities(medicine_id, 'medicine_disease')
        display_related_entities(related_diseases, diseases_dict, 'disease_id')

        # Display related side effects
        st.subheader('Related Side Effects')
        related_side_effects = find_related_entities(medicine_id, 'medicine_side_effect')
        display_related_entities(related_side_effects, side_effects_dict, 'side_effect_id')

        # Display related interactions
        st.subheader('Related Interactions')
        related_interactions = find_related_entities(medicine_id, 'medicine_interaction')
        display_related_entities(related_interactions, interactions_dict, 'interaction_id')

        # Display related warnings
        st.subheader('Related Warnings')
        related_warnings = find_related_entities(medicine_id, 'medicine_warning')
        display_related_entities(related_warnings, warnings_dict, 'warning_id')

        # Display related contraindications
        st.subheader('Related Contraindications')
        related_contraindications = find_related_entities(medicine_id, 'medicine_contraindication')
        display_related_entities(related_contraindications, contraindications_dict, 'contraindication_id')

        # Display related mechanisms of action
        st.subheader('Related Mechanisms of Action')
        related_mechanisms = find_related_entities(medicine_id, 'medicine_mechanism')
        display_related_entities(related_mechanisms, mechanisms_dict, 'mechanism_id')

elif search_type == 'Symptom':
    # Symptom Selection
    symptom_options = {sym['name']: sym['id'] for sym in symptoms}
    selected_symptom = st.selectbox('Select a Symptom', list(symptom_options.keys()))

    if selected_symptom:
        symptom_id = symptom_options[selected_symptom]
        
        # Display related medicines
        st.subheader('Related Medicines')
        related_medicines = find_related_entities(symptom_id, 'medicine_symptom', 'symptom_id')
        display_related_entities(related_medicines, medicines_dict, 'medicine_id')

elif search_type == 'Indication':
    # Indication Selection
    indication_options = {ind['name']: ind['id'] for ind in indications}
    selected_indication = st.selectbox('Select an Indication', list(indication_options.keys()))

    if selected_indication:
        indication_id = indication_options[selected_indication]

        # Display related medicines
        st.subheader('Related Medicines')
        related_medicines = find_related_entities(indication_id, 'medicine_indication', 'indication_id')
        display_related_entities(related_medicines, medicines_dict, 'medicine_id')

elif search_type == 'Disease':
    # Disease Selection
    disease_options = {dis['name']: dis['id'] for dis in diseases}
    selected_disease = st.selectbox('Select a Disease', list(disease_options.keys()))

    if selected_disease:
        disease_id = disease_options[selected_disease]

        # Display related medicines
        st.subheader('Related Medicines')
        related_medicines = find_related_entities(disease_id, 'medicine_disease', 'disease_id')
        display_related_entities(related_medicines, medicines_dict, 'medicine_id')

# RAG-based Search
st.subheader('RAG-based Search')
query = st.text_input('Enter your query')
if query:
    context = " ".join([med['description'] for med in medicines])  # Simplified context
    result = rag_model(question=query, context=context)
    st.write(result['answer'])

# Run the app
if __name__ == '__main__':
    st.write("Welcome to the Medicine Knowledge Base")
