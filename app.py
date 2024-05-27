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

# Function to inspect data structure
def inspect_data(data):
    structure = {}
    for table_name, content in data.items():
        if isinstance(content, list) and len(content) > 0:
            structure[table_name] = list(content[0].keys())
        else:
            structure[table_name] = []
    return structure

# Load all data from the 'data' folder
data = load_data('data')
data_structure = inspect_data(data)

# Display data structure for debugging
st.write("Data Structure:", data_structure)

# Function to normalize and merge data based on foreign keys
def normalize_and_merge(data, data_structure):
    df_dict = {table_name: pd.DataFrame(content) for table_name, content in data.items()}

    # Handling many-to-many relationships using intermediate tables
    for intermediate_table in [
        'generic_name_to_brand', 'generic_name_to_medicine', 'disease_to_medicine', 'symptom_to_medicine',
        'mechanism_of_action_to_medicine', 'indication_to_medicine', 'contraindication_to_medicine',
        'warning_to_medicine', 'interaction_to_medicine', 'side_effect_to_medicine'
    ]:
        if intermediate_table in df_dict:
            df = df_dict[intermediate_table]
            main_table = intermediate_table.split('_to_')[1]
            if main_table in df_dict:
                for col in df.columns:
                    if col.endswith("_id") and col[:-3] in data_structure:
                        foreign_table = col[:-3]
                        df = df.merge(df_dict[foreign_table], left_on=col, right_on="id", suffixes=('', f'_{foreign_table}')).drop(columns=[col, 'id_' + foreign_table])
                df_dict[main_table] = df_dict[main_table].merge(df, how='left', left_on='id', right_on=f'{main_table}_id').drop(columns=[f'{main_table}_id'])

    # Handling one-to-many and many-to-one relationships
    for table_name, columns in data_structure.items():
        df = df_dict[table_name]
        for col in columns:
            if col.endswith("_id") and col[:-3] in data_structure:
                foreign_table = col[:-3]
                df = df.merge(df_dict[foreign_table], left_on=col, right_on="id", suffixes=('', f'_{foreign_table}'))
                df = df.drop(columns=[col, 'id_' + foreign_table])

        df_dict[table_name] = df

    return df_dict

# Normalize and merge data
df_dict = normalize_and_merge(data, data_structure)

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
            related_medicines = df_dict["medicines"][df_dict["medicines"]["id"].isin(result["id"])]
            related_symptoms = df_dict["symptoms"][df_dict["symptoms"]["id"].isin(result["id"])]
            st.write("Related Medicines:", related_medicines)
            st.write("Related Symptoms:", related_symptoms)
        elif selected_table == "symptoms":
            related_diseases = df_dict["diseases"][df_dict["diseases"]["id"].isin(result["id"])]
            related_medicines = df_dict["medicines"][df_dict["medicines"]["id"].isin(result["id"])]
            st.write("Related Diseases:", related_diseases)
            st.write("Related Medicines:", related_medicines)
        elif selected_table == "medicines":
            related_generic_names = df_dict["generic_names"][df_dict["generic_names"]["id"].isin(result["id"])]
            related_brands = df_dict["brand_names"][df_dict["brand_names"]["id"].isin(result["id"])]
            related_indications = df_dict["indications"][df_dict["indications"]["id"].isin(result["id"])]
            related_contraindications = df_dict["contraindications"][df_dict["contraindications"]["id"].isin(result["id"])]
            related_warnings = df_dict["warnings"][df_dict["warnings"]["id"].isin(result["id"])]
            related_interactions = df_dict["interactions"][df_dict["interactions"]["id"].isin(result["id"])]
            related_side_effects = df_dict["side_effects"][df_dict["side_effects"]["id"].isin(result["id"])]
            st.write("Related Generic Names:", related_generic_names)
            st.write("Related Brands:", related_brands)
            st.write("Related Indications:", related_indications)
            st.write("Related Contraindications:", related_contraindications)
            st.write("Related Warnings:", related_warnings)
            st.write("Related Interactions:", related_interactions)
            st.write("Related Side Effects:", related_side_effects)

