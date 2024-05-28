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
            for col in df.columns:
                if col.endswith("_id") and col[:-3] in data_structure:
                    foreign_table = col[:-3]
                    df = df.merge(df_dict[foreign_table], left_on=col, right_on="id", suffixes=('', f'_{foreign_table}')).drop(columns=[col, 'id_' + foreign_table])
            if main_table in df_dict:
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
        def display_related_data(main_table, result_ids, related_table, intermediate_table):
            if intermediate_table in df_dict:
                related_ids = df_dict[intermediate_table][f'{related_table[:-1]}_id'][df_dict[intermediate_table][f'{main_table[:-1]}_id'].isin(result_ids)].unique()
                related_data = df_dict[related_table][df_dict[related_table]['id'].isin(related_ids)]
                return related_data
            return pd.DataFrame()

        if selected_table == "diseases":
            related_medicines = display_related_data("diseases", result["id"], "medicines", "disease_to_medicine")
            related_symptoms = display_related_data("medicines", related_medicines["id"], "symptoms", "symptom_to_medicine")
            st.write("Related Medicines:", related_medicines)
            st.write("Related Symptoms:", related_symptoms)
        elif selected_table == "symptoms":
            related_medicines = display_related_data("symptoms", result["id"], "medicines", "symptom_to_medicine")
            related_diseases = display_related_data("medicines", related_medicines["id"], "diseases", "disease_to_medicine")
            st.write("Related Diseases:", related_diseases)
            st.write("Related Medicines:", related_medicines)
        elif selected_table == "medicines":
            related_generic_names = display_related_data("medicines", result["id"], "generic_names", "generic_name_to_medicine")
            related_brands = df_dict["brand_names"][df_dict["brand_names"]["id"].isin(result["brand_id"])]
            related_indications = display_related_data("medicines", result["id"], "indications", "indication_to_medicine")
            related_contraindications = display_related_data("medicines", result["id"], "contraindications", "contraindication_to_medicine")
            related_warnings = display_related_data("medicines", result["id"], "warnings", "warning_to_medicine")
            related_interactions = display_related_data("medicines", result["id"], "interactions", "interaction_to_medicine")
            related_side_effects = display_related_data("medicines", result["id"], "side_effects", "side_effect_to_medicine")
            related_mechanisms = display_related_data("medicines", result["id"], "mechanism_of_action", "mechanism_of_action_to_medicine")
            st.write("Related Generic Names:", related_generic_names)
            st.write("Related Brands:", related_brands)
            st.write("Related Indications:", related_indications)
            st.write("Related Contraindications:", related_contraindications)
            st.write("Related Warnings:", related_warnings)
            st.write("Related Interactions:", related_interactions)
            st.write("Related Side Effects:", related_side_effects)
            st.write("Related Mechanisms of Action:", related_mechanisms)
