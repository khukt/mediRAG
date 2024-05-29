import json

# Load JSON data
with open('medicines.json') as f:
    medicines = json.load(f)['medicines']
with open('symptoms.json') as f:
    symptoms = json.load(f)['symptoms']
with open('relationships.json') as f:
    relationships = json.load(f)

# Convert to dictionaries for easy lookup
medicines_dict = {med['id']: med for med in medicines}
symptoms_dict = {sym['id']: sym for sym in symptoms}
medicine_symptom_rel = relationships['medicine_symptom']

# Function to find symptoms for a given medicine ID
def find_symptoms_for_medicine(medicine_id):
    related_symptoms = [rel['symptom_id'] for rel in medicine_symptom_rel if rel['medicine_id'] == medicine_id]
    return [symptoms_dict[sym_id] for sym_id in related_symptoms]

# Example usage
medicine_id = 1
symptoms = find_symptoms_for_medicine(medicine_id)
print(f"Symptoms related to medicine ID {medicine_id}:")
for symptom in symptoms:
    print(symptom)
