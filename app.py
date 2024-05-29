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
generic_names = load_data('generic_names.json')['generic_names']

# Convert to dictionaries for easy lookup
def to_dict(data, key='id'):
    return {item[key]: item for item in data}

medicines_dict = to_dict(medicines)
symptoms_dict = to_dict(symptoms)
indications_dict = to_dict(indications)
generic_names_dict = to_dict(generic_names)

# Load RAG model
@st.cache(allow_output_mutation=True)
def load_rag_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

rag_model = load_rag_model()

# Function to handle specific queries
def handle_specific_query(query):
    # Check for direct medicine queries
    medicine_name_lookup = {med['name'].lower(): med for med in medicines}
    
    if "compare" in query.lower():
        # Handle comparison queries
        meds = [med.strip().lower() for med in query.lower().replace("compare", "").split("and")]
        if len(meds) == 2 and meds[0] in medicine_name_lookup and meds[1] in medicine_name_lookup:
            med1 = medicine_name_lookup[meds[0]]
            med2 = medicine_name_lookup[meds[1]]
            return f"Comparison between {med1['name']} and {med2['name']}:\n\n{med1['description']}\n\n{med2['description']}"
    
    if query.lower() in medicine_name_lookup:
        medicine = medicine_name_lookup[query.lower()]
        return f"**Name:** {medicine['name']}\n**Description:** {medicine['description']}"

    if "generic name of" in query.lower():
        brand = query.lower().replace("generic name of", "").strip()
        for med in medicines:
            if med['name'].lower() == brand:
                generic_id = med.get('generic_id')
                if generic_id:
                    generic_name = generic_names_dict[generic_id]['name']
                    return f"The generic name of {brand} is {generic_name}."
                else:
                    return f"Generic name information is not available for {brand}."
    
    return None

# Streamlit UI
st.title('Medicine Knowledge Base with Advanced Search')

# RAG-based Search
st.subheader('RAG-based Search')
query = st.text_input('Enter your query')
if query:
    # Check for specific queries first
    specific_answer = handle_specific_query(query)
    if specific_answer:
        st.write(specific_answer)
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

# Run the app
if __name__ == '__main__':
    st.write("Welcome to the Medicine Knowledge Base")
