import streamlit as st
import json
import pandas as pd
import os

# Function to load data from JSON files dynamically
def load_data(folder):
    data = {}
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            table_name = filename.replace(".json", "")
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                data[table_name] = json.load(file)
    return data

# Load all data from the 'data' folder
data = load_data('data')

# Function to convert a list of dictionaries to a DataFrame
def list_to_dataframe(data, key):
    if key in data:
        return pd.DataFrame(data[key])
    return pd.DataFrame()

# Convert all data to DataFrames
df_dict = {key: list_to_dataframe(data, key) for key in data}

# Function to handle one-to-one relationships
def handle_one_to_one(df, main_key, related_key):
    if main_key in df_dict and related_key in df_dict:
        return df_dict[main_key].merge(df_dict[related_key], how='left', left_on='id', right_on=f'{main_key}_id')

# Function to handle many-to-one relationships
def handle_many_to_one(df, main_key, related_key):
    if main_key in df_dict and related_key in df_dict:
        return df_dict[main_key].merge(df_dict[related_key], how='left', left_on=f'{related_key}_id', right_on='id')

# Function to handle one-to-many relationships
def handle_one_to_many(df, main_key, related_key):
    if main_key in df_dict and related_key in df_dict:
        return df_dict[main_key].merge(df_dict[related_key], how='left', left_on='id', right_on=f'{main_key}_id')

# Function to handle many-to-many relationships using intermediate tables
def handle_many_to_many(intermediate_table, main_table, related_table):
    if intermediate_table in df_dict:
        df = df_dict[intermediate_table]
        for col in df.columns:
            if col.endswith("_id"):
                related_df = df_dict[col[:-3]]
                df = df.merge(related_df, left_on=col, right_on="id", suffixes=('', f'_{col[:-3]}')).drop(columns=[col])
        if main_table in df_dict:
            df_dict[main_table] = df_dict[main_table].merge(df, how='left', left_on='id', right_on=f'{main_table}_id').drop(columns=[f'{main_table}_id'])

# Handle relationships
relationships = [
    ('generic_name_to_medicine', 'medicines', 'generic_names'),
    ('disease_to_medicine', 'diseases', 'medicines'),
    ('symptom_to_medicine', 'symptoms', 'medicines'),
    ('mechanism_of_action_to_medicine', 'medicines', 'mechanism_of_action'),
    ('indication_to_medicine', 'medicines', 'indications'),
    ('contraindication_to_medicine', 'medicines', 'contraindications'),
    ('warning_to_medicine', 'medicines', 'warnings'),
    ('interaction_to_medicine', 'medicines', 'interactions'),
    ('side_effect_to_medicine', 'medicines', 'side_effects'),
]

for rel in relationships:
    handle_many_to_many(*rel)

# Streamlit interface for displaying data
st.title("MediRAG Data Viewer")

# Dropdown to select the main table
selected_table = st.selectbox("Select Table", list(df_dict.keys()))

# Display the selected table's data
if selected_table:
    st.subheader(f"{selected_table.capitalize()} Data")
    st.write(df_dict[selected_table])

    # Option to select a specific row
    row_id = st.number_input(f"Select {selected_table} ID", min_value=0, max_value=len(df_dict[selected_table])-1, step=1)

    if row_id:
        result = df_dict[selected_table].iloc[row_id]
        st.write(result)

        # Display related data based on the table selected
        if selected_table == "diseases":
            related_medicines = handle_many_to_many('disease_to_medicine', 'diseases', 'medicines')
            related_symptoms = handle_many_to_many('symptom_to_medicine', 'diseases', 'symptoms')
            st.write("Related Medicines:", related_medicines)
            st.write("Related Symptoms:", related_symptoms)
        elif selected_table == "symptoms":
            related_medicines = handle_many_to_many('symptom_to_medicine', 'symptoms', 'medicines')
            st.write("Related Medicines:", related_medicines)
        elif selected_table == "medicines":
            related_generic_names = handle_many_to_many('generic_name_to_medicine', 'medicines', 'generic_names')
            related_brands = df_dict["brand_names"][df_dict["brand_names"]["id"].isin(result["brand_id"])]
            related_indications = handle_many_to_many('indication_to_medicine', 'medicines', 'indications')
            related_contraindications = handle_many_to_many('contraindication_to_medicine', 'medicines', 'contraindications')
            related_warnings = handle_many_to_many('warning_to_medicine', 'medicines', 'warnings')
            related_interactions = handle_many_to_many('interaction_to_medicine', 'medicines', 'interactions')
            related_side_effects = handle_many_to_many('side_effect_to_medicine', 'medicines', 'side_effects')
            related_mechanisms = handle_many_to_many('mechanism_of_action_to_medicine', 'medicines', 'mechanism_of_action')
            st.write("Related Generic Names:", related_generic_names)
            st.write("Related Brands:", related_brands)
            st.write("Related Indications:", related_indications)
            st.write("Related Contraindications:", related_contraindications)
            st.write("Related Warnings:", related_warnings)
            st.write("Related Interactions:", related_interactions)
            st.write("Related Side Effects:", related_side_effects)
            st.write("Related Mechanisms of Action:", related_mechanisms)
