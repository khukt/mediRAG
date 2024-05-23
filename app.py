import streamlit as st
import json

@st.cache
def load_data():
    with open('brand_names.json') as f:
        brand_names = json.load(f)['brand_names']
    with open('diseases.json') as f:
        diseases = json.load(f)['diseases']
    with open('forms.json') as f:
        forms = json.load(f)['forms']
    with open('generic_names.json') as f:
        generic_names = json.load(f)['generic_names']
    with open('manufacturers.json') as f:
        manufacturers = json.load(f)['manufacturers']
    with open('medicines.json') as f:
        medicines = json.load(f)['medicines']
    with open('symptoms.json') as f:
        symptoms = json.load(f)['symptoms']
    
    return brand_names, diseases, forms, generic_names, manufacturers, medicines, symptoms

brand_names, diseases, forms, generic_names, manufacturers, medicines, symptoms = load_data()

# Create index dictionaries for fast lookups
index_data = {
    "medicines": {item['description'].lower(): item for item in medicines},
    "diseases": {item['name'].lower(): item for item in diseases},
    "symptoms": {item['name'].lower(): item for item in symptoms},
    "manufacturers": {item['name'].lower(): item for item in manufacturers},
    "brand_names": {item['name'].lower(): item for item in brand_names}
}

def search_data(query):
    query_lower = query.lower()
    results = {
        "medicines": [],
        "diseases": [],
        "symptoms": [],
        "manufacturers": [],
        "brand_names": []
    }
    
    # Search in each index dictionary
    for category, index in index_data.items():
        for key in index:
            if query_lower in key:
                results[category].append(index[key])
    
    return results

st.title("MediRAG - Medicine Information System")

st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Home", "Search"])

if options == "Home":
    st.write("Welcome to the MediRAG Medicine Information System.")
elif options == "Search":
    st.header("Search")
    query = st.text_input("Enter a medicine name, disease, symptom, or manufacturer")
    
    if query:
        results = search_data(query)
        
        st.subheader("Medicines")
        for med in results["medicines"]:
            st.write(f"**{med['description']}**")
            st.write(f"Indications: {', '.join(med['indications'])}")
            st.write(f"Warnings: {', '.join(med['warnings'])}")
            st.write(f"Side Effects: {', '.join(med['side_effects'])}")
        
        st.subheader("Diseases")
        for disease in results["diseases"]:
            st.write(f"**{disease['name']}**")
        
        st.subheader("Symptoms")
        for symptom in results["symptoms"]:
            st.write(f"**{symptom['name']}**")
        
        st.subheader("Manufacturers")
        for manufacturer in results["manufacturers"]:
            st.write(f"**{manufacturer['name']}**")
            st.write(f"Contact Info: {manufacturer['contact_info']}")
        
        st.subheader("Brand Names")
        for brand in results["brand_names"]:
            st.write(f"**{brand['name']}**")
