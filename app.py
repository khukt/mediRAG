import streamlit as st
import json

# Load JSON data
@st.cache
def load_data(file):
    with open(file) as f:
        return json.load(f)

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
medicines_dict = {med['id']: med for med in medicines}
symptoms_dict = {sym['id']: sym for sym in symptoms}
indications_dict = {ind['id']: ind for ind in indications}
diseases_dict = {dis['id']: dis for dis in diseases}
side_effects_dict = {se['id']: se for se in side_effects}
interactions_dict = {intc['id']: intc for intc in interactions}
warnings_dict = {war['id']: war for war in warnings}
contraindications_dict = {con['id']: con for con in contraindications}
mechanisms_dict = {moa['id']: moa for moa in mechanisms_of_action}
brand_names_dict = {bn['id']: bn for bn in brand_names}
generic_names_dict = {gn['id']: gn for gn in generic_names}
manufacturers_dict = {mf['id']: mf for mf in manufacturers}

# Function to find relationships
def find_related_entities(medicine_id, relationship):
    related_ids = [rel for rel in relationships[relationship] if rel['medicine_id'] == medicine_id]
    return related_ids

# Streamlit UI
st.title('Medicine Knowledge Base')

# Medicine Selection
medicine_options = {med['name']: med['id'] for med in medicines}
selected_medicine = st.selectbox('Select a Medicine', list(medicine_options.keys()))

if selected_medicine:
    medicine_id = medicine_options[selected_medicine]
    st.write(f"**Description:** {medicines_dict[medicine_id]['description']}")

    # Display related symptoms
    st.subheader('Related Symptoms')
    related_symptoms = find_related_entities(medicine_id, 'medicine_symptom')
    for rel in related_symptoms:
        st.write(symptoms_dict[rel['symptom_id']]['name'])

    # Display related indications
    st.subheader('Related Indications')
    related_indications = find_related_entities(medicine_id, 'medicine_indication')
    for rel in related_indications:
        st.write(indications_dict[rel['indication_id']]['name'])

    # Display related diseases
    st.subheader('Related Diseases')
    related_diseases = find_related_entities(medicine_id, 'medicine_disease')
    for rel in related_diseases:
        st.write(diseases_dict[rel['disease_id']]['name'])

    # Display related side effects
    st.subheader('Related Side Effects')
    related_side_effects = find_related_entities(medicine_id, 'medicine_side_effect')
    for rel in related_side_effects:
        st.write(side_effects_dict[rel['side_effect_id']]['name'])

    # Display related interactions
    st.subheader('Related Interactions')
    related_interactions = find_related_entities(medicine_id, 'medicine_interaction')
    for rel in related_interactions:
        st.write(interactions_dict[rel['interaction_id']]['name'])

    # Display related warnings
    st.subheader('Related Warnings')
    related_warnings = find_related_entities(medicine_id, 'medicine_warning')
    for rel in related_warnings:
        st.write(warnings_dict[rel['warning_id']]['name'])

    # Display related contraindications
    st.subheader('Related Contraindications')
    related_contraindications = find_related_entities(medicine_id, 'medicine_contraindication')
    for rel in related_contraindications:
        st.write(contraindications_dict[rel['contraindication_id']]['name'])

    # Display related mechanisms of action
    st.subheader('Related Mechanisms of Action')
    related_mechanisms = find_related_entities(medicine_id, 'medicine_mechanism')
    for rel in related_mechanisms:
        st.write(mechanisms_dict[rel['mechanism_id']]['description'])

# Run the app
if __name__ == '__main__':
    st.write("Welcome to the Medicine Knowledge Base")
