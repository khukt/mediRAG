import streamlit as st
import json
import pandas as pd

# Function to load data from JSON files
def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

# Load all data
symptoms = load_json('symptoms.json')["symptoms"]
generic_names = load_json('generic_names.json')["generic_names"]
diseases = load_json('diseases.json')["diseases"]
forms = load_json('forms.json')["forms"]
brand_names = load_json('brand_names.json')["brand_names"]
manufacturers = load_json('manufacturers.json')["manufacturers"]
medicines = load_json('medicines.json')["medicines"]

# Convert lists to DataFrames for easier manipulation
df_symptoms = pd.DataFrame(symptoms)
df_generic_names = pd.DataFrame(generic_names)
df_diseases = pd.DataFrame(diseases)
df_forms = pd.DataFrame(forms)
df_brand_names = pd.DataFrame(brand_names)
df_manufacturers = pd.DataFrame(manufacturers)
df_medicines = pd.DataFrame(medicines)

# Merge brands with manufacturers
df_brand_names = df_brand_names.merge(df_manufacturers, left_on='manufacturer_id', right_on='id', suffixes=('', '_manufacturer')).drop(columns=['manufacturer_id'])

# Expand brands in medicines
df_medicines = df_medicines.explode('brands').reset_index(drop=True)

# Normalize nested brand information in medicines
brands_normalized = pd.json_normalize(df_medicines['brands'])
df_medicines = df_medicines.drop(columns=['brands']).join(brands_normalized)

# Merge brand names into medicines
df_medicines = df_medicines.merge(df_brand_names, left_on='brand_id', right_on='id', suffixes=('', '_brand')).drop(columns=['brand_id', 'id_brand'])

# Merge forms into medicines
df_medicines = df_medicines.merge(df_forms, left_on='form_id', right_on='id', suffixes=('', '_form')).drop(columns(['form_id', 'id_form']))

# Merge generic names into medicines
df_medicines = df_medicines.explode('generic_name_ids').merge(df_generic_names, left_on='generic_name_ids', right_on='id', suffixes=('', '_generic')).drop(columns(['generic_name_ids', 'id_generic']))

# Merge symptoms into medicines
df_medicines = df_medicines.explode('symptom_ids').merge(df_symptoms, left_on='symptom_ids', right_on='id', suffixes=('', '_symptom')).drop(columns(['symptom_ids', 'id_symptom']))

# Merge diseases into medicines
df_medicines = df_medicines.explode('disease_ids').merge(df_diseases, left_on='disease_ids', right_on='id', suffixes=('', '_disease')).drop(columns(['disease_ids', 'id_disease']))

# Function to create combined text for medicines
def create_combined_text(row):
    fields = [
        'description', 'description_mm', 'mechanism_of_action', 'mechanism_of_action_mm', 
        'indications', 'indications_mm', 'contraindications', 'contraindications_mm',
        'warnings', 'warnings_mm', 'interactions', 'interactions_mm', 'side_effects',
        'side_effects_mm', 'name', 'name_generic', 'name_brand', 'name_symptom',
        'name_disease', 'additional_info', 'additional_info_mm'
    ]
    combined_text = ' '.join([str(row[field]) for field in fields if field in row])
    return combined_text

# Add combined text column
df_medicines['combined_text'] = df_medicines.apply(create_combined_text, axis=1)

# Function to retrieve data by medicine ID
def get_medicine_by_id(medicine_id):
    result = df_medicines[df_medicines['id'] == medicine_id]
    return result.to_dict(orient='records')

# Function to retrieve medicines by symptom
def get_medicines_by_symptom(symptom_name):
    result = df_medicines[df_medicines['name_symptom'].str.contains(symptom_name, case=False)]
    return result.drop_duplicates().to_dict(orient='records')

# Function to retrieve medicines by disease
def get_medicines_by_disease(disease_name):
    result = df_medicines[df_medicines['name_disease'].str.contains(disease_name, case=False)]
    return result.drop_duplicates().to_dict(orient='records')

# Function to retrieve medicines by generic name
def get_medicines_by_generic_name(generic_name):
    result = df_medicines[df_medicines['name_generic'].str.contains(generic_name, case=False)]
    return result.drop_duplicates().to_dict(orient='records')

# Function to retrieve medicines by brand name
def get_medicines_by_brand_name(brand_name):
    result = df_medicines[df_medicines['name_brand'].str.contains(brand_name, case=False)]
    return result.drop_duplicates().to_dict(orient='records')

# Function to retrieve all details for a specific medicine
def get_medicine_details(medicine_id):
    medicine_details = get_medicine_by_id(medicine_id)
    if not medicine_details:
        return "No medicine found with the given ID."
    return medicine_details

# Streamlit App
st.title("Medicine Information Retrieval")

st.write("Use the options below to query the medicine database:")

# Sidebar options
query_type = st.sidebar.selectbox("Select Query Type", ["By ID", "By Symptom", "By Disease", "By Generic Name", "By Brand Name"])

if query_type == "By ID":
    medicine_id = st.sidebar.number_input("Enter Medicine ID", min_value=1)
    if st.sidebar.button("Search"):
        result = get_medicine_by_id(medicine_id)
        st.write(result)

elif query_type == "By Symptom":
    symptom_name = st.sidebar.text_input("Enter Symptom Name")
    if st.sidebar.button("Search"):
        result = get_medicines_by_symptom(symptom_name)
        st.write(result)

elif query_type == "By Disease":
    disease_name = st.sidebar.text_input("Enter Disease Name")
    if st.sidebar.button("Search"):
        result = get_medicines_by_disease(disease_name)
        st.write(result)

elif query_type == "By Generic Name":
    generic_name = st.sidebar.text_input("Enter Generic Name")
    if st.sidebar.button("Search"):
        result = get_medicines_by_generic_name(generic_name)
        st.write(result)

elif query_type == "By Brand Name":
    brand_name = st.sidebar.text_input("Enter Brand Name")
    if st.sidebar.button("Search"):
        result = get_medicines_by_brand_name(brand_name)
        st.write(result)
