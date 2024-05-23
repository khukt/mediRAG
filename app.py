import streamlit as st
import json

# Load the databases
with open('medicines.json') as f:
    medicines = json.load(f)['medicines']

with open('brand_names.json') as f:
    brand_names = json.load(f)['brand_names']

with open('manufacturers.json') as f:
    manufacturers = json.load(f)['manufacturers']

# Create dictionaries for quick lookups
brand_dict = {brand['id']: brand for brand in brand_names}
manufacturer_dict = {manufacturer['id']: manufacturer for manufacturer in manufacturers}

# Function to retrieve medicine information
def get_medicine_info(query):
    query = query.lower()
    results = []
    for med in medicines:
        if query in med['generic_name'].lower() or any(query in brand_dict[brand_id]['name'].lower() for brand_id in med['brand_names']) or any(query in use.lower() for use in med['uses']):
            results.append(med)
    return results

# Streamlit app
st.title('Medicine Information Retrieval')

st.write('Enter a medicine name (generic or brand) or a symptom to get information about relevant medicines.')

query = st.text_input('Query')

if query:
    results = get_medicine_info(query)
    if results:
        for med in results:
            st.subheader(f"Generic Name: {med['generic_name']}")
            st.write('**Uses:**', ', '.join(med['uses']))
            st.write('**Side Effects:**', ', '.join(med['side_effects']))
            for brand_id in med['brand_names']:
                brand = brand_dict[brand_id]
                manufacturer = manufacturer_dict[brand['manufacturer_id']]
                st.write(f"**Brand Name:** {brand['name']}")
                st.write(f"**Dosages:** {', '.join(brand['dosages'])}")
                st.write(f"**Manufacturer:** {manufacturer['name']}")
                st.write(f"**Contact Info:** Phone: {manufacturer['contact_info']['phone']}, Email: {manufacturer['contact_info']['email']}, Address: {manufacturer['contact_info']['address']}")
    else:
        st.write('No results found.')

