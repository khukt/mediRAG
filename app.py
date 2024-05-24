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

    # Merging based on assumed foreign key convention (e.g., "table_id")
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
st.title("Dynamic Data Retrieval App")

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
        st.write(result)

# Function to replace IDs with real names
def replace_ids_with_names(df, data_structure, df_dict):
    for col in df.columns:
        if col.endswith("_id") and col[:-3] in data_structure:
            foreign_table = col[:-3]
            foreign_df = df_dict[foreign_table].set_index('id')
            df[col] = df[col].map(foreign_df['name'])
    return df

# Replace IDs with real names in all DataFrames
df_dict = {table_name: replace_ids_with_names(df, data_structure, df_dict) for table_name, df in df_dict.items()}
