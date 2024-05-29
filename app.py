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
relationships = load_data('relationships.json')

# Convert to dictionaries for easy lookup
def to_dict(data, key='id'):
    return {item[key]: item for item in data}

medicines_dict = to_dict(medicines)
symptoms_dict = to_dict(symptoms)
indications_dict = to_dict(indications)

# Load RAG model
@st.cache(allow_output_mutation=True)
def load_rag_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

rag_model = load_rag_model()

# Streamlit UI
st.title('Medicine Knowledge Base with RAG-based Search')

# RAG-based Search
st.subheader('RAG-based Search')
query = st.text_input('Enter your query')
if query:
    # Direct lookup for specific medicine names
    medicine_name_lookup = {med['name'].lower(): med for med in medicines}
    if query.lower() in medicine_name_lookup:
        medicine = medicine_name_lookup[query.lower()]
        st.write(f"**Name:** {medicine['name']}")
        st.write(f"**Description:** {medicine['description']}")

        # Suggest alternative medicines based on symptoms and indications
        symptom_ids = [rel['symptom_id'] for rel in relationships['medicine_symptom'] if rel['medicine_id'] == medicine['id']]
        indication_ids = [rel['indication_id'] for rel in relationships['medicine_indication'] if rel['medicine_id'] == medicine['id']]

        alternative_medicines = set()
        for symptom_id in symptom_ids:
            alternative_medicines.update(find_medicines_by_symptom(symptom_id))
        for indication_id in indication_ids:
            alternative_medicines.update(find_medicines_by_indication(indication_id))
        alternative_medicines.discard(medicine['id'])  # Remove the current medicine from the alternatives

        st.subheader('Alternative Medicines')
        for med_id in alternative_medicines:
            st.write(medicines_dict[med_id]['name'])

    else:
        # Create a focused context for the query
        context = ""
        for medicine in medicines:
            context += f"Medicine: {medicine['name']}\nDescription: {medicine['description']}\n\n"
        for symptom in symptoms:
            context += f"Symptom: {symptom['name']}\n\n"
        for indication in indications:
            context += f"Indication: {indication['name']}\n\n"
        
        result = rag_model(question=query, context=context)
        st.write(result['answer'])

        # Suggest alternative medicines based on symptoms and indications in the query
        for symptom in symptoms:
            if symptom['name'].lower() in query.lower():
                symptom_id = symptom['id']
                alternative_medicines_ids = find_medicines_by_symptom(symptom_id)
                alternative_medicines = [medicines_dict[med_id]['name'] for med_id in alternative_medicines_ids if med_id in medicines_dict]
                
                st.subheader('Alternative Medicines')
                st.write(f"For symptom: {symptom['name']}")
                for med in alternative_medicines:
                    st.write(med)
                break

        for indication in indications:
            if indication['name'].lower() in query.lower():
                indication_id = indication['id']
                alternative_medicines_ids = find_medicines_by_indication(indication_id)
                alternative_medicines = [medicines_dict[med_id]['name'] for med_id in alternative_medicines_ids if med_id in medicines_dict]
                
                st.subheader('Alternative Medicines')
                st.write(f"For indication: {indication['name']}")
                for med in alternative_medicines:
                    st.write(med)
                break

# Run the app
if __name__ == '__main__':
    st.write("Welcome to the Medicine Knowledge Base")
