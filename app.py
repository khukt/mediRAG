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

# Display data structure for debugging
st.write("Loaded Data:", data)

# Function to convert a list of dictionaries to a DataFrame
def list_to_dataframe(data, key):
    if key in data:
        return pd.DataFrame(data[key])
    return pd.DataFrame()

# Convert all data to DataFrames
df_dict = {key: list_to_dataframe(data, key) for key in data}

# Function to handle many-to-many relationships using intermediate tables
def handle_many_to_many(df_dict, intermediate_table, main_table, related_table):
    if intermediate_table in df_dict:
        df = df_dict[intermediate_table]
        for col in df.columns:
            if col.endswith("_id"):
                related_df = df_dict[col[:-3]]
                df = df.merge(related_df, left_on=col, right_on="id", suffixes=('', f'_{col[:-3]}')).drop(columns=[col])
        df_dict[main_table] = df_dict[main_table].merge(df, how='left', left_on='id', right_on=f'{main_table}_id').drop(columns=[f'{main_table}_id'])

# Handle many-to-many relationships
for intermediate_table, main_table, related_table in [
    ('generic_name_to_medicine', 'medicines', 'generic_names'),
    ('disease_to_medicine', 'diseases', 'medicines'),
    ('symptom_to_medicine', 'symptoms', 'medicines'),
    ('mechanism_of_action_to_medicine', 'medicines', 'mechanism_of_action'),
    ('indication_to_medicine', 'medicines', 'indications'),
    ('contraindication_to_medicine', 'medicines', 'contraindications'),
    ('warning_to_medicine', 'medicines', 'warnings'),
    ('interaction_to_medicine', 'medicines', 'interactions'),
    ('side_effect_to_medicine', 'medicines', 'side_effects')
]:
    handle_many_to_many(df_dict, intermediate_table, main_table, related_table)

# Display normalized and merged data for debugging
st.write("Normalized and Merged Data:", df_dict)

# Function to retrieve related data
def get_related_data(main_table, result_ids, related_table, intermediate_table):
    if intermediate_table in df_dict:
        related_ids = df_dict[intermediate_table][f'{related_table[:-1]}_id'][df_dict[intermediate_table][f'{main_table[:-1]}_id'].isin(result_ids)].unique()
        related_data = df_dict[related_table][df_dict[related_table]['id'].isin(related_ids)]
        return related_data
    return pd.DataFrame()

# Streamlit app to query data dynamically
st.title("Enhanced Dynamic Data Retrieval App")

# Select table to query
selected_table = st.selectbox("Select Table", list(df_dict.keys()))

if selected_table:
    df = df_dict[selected_table]
    st.write(f"Columns in {selected_table}:", df.columns.tolist())

    # Select column to filter
    selected_column = st.selectbox("Select Column to Filter", df.columns)
    filter_value = st.text_input(f"Enter value to filter {selected_column}")

    if st.button("Search"):
        result = df[df[selected_column].astype(str).str.contains(filter_value, case=False, na=False)]
        st.write("Search Results:", result)

        # Display related data
        if selected_table == "diseases":
            related_medicines = get_related_data("diseases", result["id"], "medicines", "disease_to_medicine")
            related_symptoms = get_related_data("medicines", related_medicines["id"], "symptoms", "symptom_to_medicine")
            st.write("Related Medicines:", related_medicines)
            st.write("Related Symptoms:", related_symptoms)
        elif selected_table == "symptoms":
            related_medicines = get_related_data("symptoms", result["id"], "medicines", "symptom_to_medicine")
            related_diseases = get_related_data("medicines", related_medicines["id"], "diseases", "disease_to_medicine")
            st.write("Related Diseases:", related_diseases)
            st.write("Related Medicines:", related_medicines)
        elif selected_table == "medicines":
            related_generic_names = get_related_data("medicines", result["id"], "generic_names", "generic_name_to_medicine")
            related_brands = df_dict["brand_names"][df_dict["brand_names"]["id"].isin(result["brand_id"])]
            related_indications = get_related_data("medicines", result["id"], "indications", "indication_to_medicine")
            related_contraindications = get_related_data("medicines", result["id"], "contraindications", "contraindication_to_medicine")
            related_warnings = get_related_data("medicines", result["id"], "warnings", "warning_to_medicine")
            related_interactions = get_related_data("medicines", result["id"], "interactions", "interaction_to_medicine")
            related_side_effects = get_related_data("medicines", result["id"], "side_effects", "side_effect_to_medicine")
            related_mechanisms = get_related_data("medicines", result["id"], "mechanism_of_action", "mechanism_of_action_to_medicine")
            st.write("Related Generic Names:", related_generic_names)
            st.write("Related Brands:", related_brands)
            st.write("Related Indications:", related_indications)
            st.write("Related Contraindications:", related_contraindications)
            st.write("Related Warnings:", related_warnings)
            st.write("Related Interactions:", related_interactions)
            st.write("Related Side Effects:", related_side_effects)
            st.write("Related Mechanisms of Action:", related_mechanisms)
