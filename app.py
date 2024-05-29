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

# Load T5 model
@st.cache(allow_output_mutation=True)
def load_t5_model():
    return pipeline("text2text-generation", model="t5-small")

t5_model = load_t5_model()

# Streamlit UI
st.title('Advanced Medicine Knowledge Base with T5')

# RAG-based Search
st.subheader('Ask a Question')
query = st.text_input('Enter your query')
if query:
    context = " ".join([med['description'] for med in medicines])  # Simplified context
    input_text = f"question: {query} context: {context}"
    response = t5_model(input_text, max_length=150, num_return_sequences=1)
    st.write(response[0]['generated_text'])

# Run the app
if __name__ == '__main__':
    st.write("Welcome to the Medicine Knowledge Base")
