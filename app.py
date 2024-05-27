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
