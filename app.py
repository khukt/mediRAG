import streamlit as st
import json
from transformers import pipeline

# Set page configuration
st.set_page_config(page_title="Advanced Medicine Knowledge Base", page_icon="💊")

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

# Function to find relationships
def find_related_entities(entity_id, relationship, entity_key='medicine_id'):
    return [rel for rel in relationships[relationship] if rel[entity_key] == entity_id]

# Function to create context from the data
def create_context():
    context = []
    for med in medicines:
        med_info = f"Medicine: {med['name']}\nDescription: {med['description']}\n"
        med_info += "Indications: " + ", ".join([indications_dict[ind['indication_id']]['name'] for ind in find_related_entities(med['id'], 'medicine_indication')]) + "\n"
        med_info += "Symptoms: " + ", ".join([symptoms_dict[sym['symptom_id']]['name'] for sym in find_related_entities(med['id'], 'medicine_symptom')]) + "\n"
        med_info += "Diseases: " + ", ".join([diseases_dict[dis['disease_id']]['name'] for dis in find_related_entities(med['id'], 'medicine_disease')]) + "\n"
        med_info += "Side Effects: " + ", ".join([side_effects_dict[se['side_effect_id']]['name'] for se in find_related_entities(med['id'], 'medicine_side_effect')]) + "\n"
        med_info += "Interactions: " + ", ".join([interactions_dict[intc['interaction_id']]['name'] for intc in find_related_entities(med['id'], 'medicine_interaction')]) + "\n"
        med_info += "Warnings: " + ", ".join([warnings_dict[warn['warning_id']]['name'] for warn in find_related_entities(med['id'], 'medicine_warning')]) + "\n"
        med_info += "Contraindications: " + ", ".join([contraindications_dict[con['contraindication_id']]['name'] for con in find_related_entities(med['id'], 'medicine_contraindication')]) + "\n"
        med_info += "Mechanisms of Action: " + ", ".join([mechanisms_dict[moa['mechanism_id']]['description'] for moa in find_related_entities(med['id'], 'medicine_mechanism')]) + "\n"
        med_info += "Brand Names: " + ", ".join([brand_names_dict[bn['id']]['name'] for bn in brand_names if bn['id'] == med.get('brand_name_id', '')]) + "\n"
        med_info += "Generic Names: " + ", ".join([generic_names_dict[gn['id']]['name'] for gn in generic_names if gn['id'] == med.get('generic_name_id', '')]) + "\n"
        med_info += "Manufacturers: " + ", ".join([manufacturers_dict[mf['id']]['name'] for mf in manufacturers if mf['id'] == med.get('manufacturer_id', '')]) + "\n"
        context.append(med_info)
    return " ".join(context)

# Load language model
@st.cache(allow_output_mutation=True)
def load_language_model():
    return pipeline("text-generation", model="distilgpt2")

language_model = load_language_model()

# Streamlit UI
st.title('💊 Advanced Medicine Knowledge Base')
st.write("""
Welcome to the Advanced Medicine Knowledge Base! Use the search functionality below to ask questions and get human-like answers based on our comprehensive database.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
search_type = st.sidebar.selectbox('Search by', ['Ask a Question', 'Medicine', 'Symptom', 'Indication', 'Disease', 'Brand Name', 'Generic Name'])

# Main Content Area
if search_type == 'Ask a Question':
    st.header('Ask a Question')
    st.write("Enter your question below and get a detailed answer based on the medicine database.")
    query = st.text_input('Enter your query')
    if query:
        context = create_context()  # Create the context based on the pre-processed data
        with st.spinner('Generating response...'):
            response = language_model(f"Context: {context}\nQuestion: {query}\nAnswer:", max_length=150, num_return_sequences=1)
        st.success("Answer:")
        st.write(f"**Question:** {query}")
        st.write(f"**Answer:** {response[0]['generated_text'].replace('Question:', '').strip()}")

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
   
