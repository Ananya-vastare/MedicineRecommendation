from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load datasets
training = pd.read_csv("Training.csv")
severity = pd.read_csv("Symptom-severity.csv")
medications = pd.read_csv("medications.csv")

# Symptoms from severity dataset - exact case
all_symptoms = severity["Symptom"].tolist()
symptom_weight = dict(zip(severity["Symptom"], severity["weight"]))


# Create a mapping from lowercase symptom (no spaces/underscores) to exact symptom name for matching user input
def normalize_symptom(symptom):
    return symptom.replace(" ", "").replace("_", "").lower()


symptom_map = {normalize_symptom(s): s for s in all_symptoms}


def convert_row_to_vector(row):
    vec = [0] * len(all_symptoms)
    for symptom in row.index:
        if symptom in all_symptoms and pd.notna(row[symptom]):
            val = str(row[symptom]).lower()
            if val == "1" or val == "yes":
                vec[all_symptoms.index(symptom)] = symptom_weight.get(symptom, 0)
    return vec


# Prepare features and labels
X = training.drop("prognosis", axis=1).apply(
    convert_row_to_vector, axis=1, result_type="expand"
)
y = training["prognosis"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)


def predict_disease_and_medicine(user_symptoms):
    # Normalize user symptoms to map to exact symptom names
    normalized = [normalize_symptom(sym) for sym in user_symptoms]
    matched_symptoms = [symptom_map[n] for n in normalized if n in symptom_map]

    # Create input vector
    user_vector = [
        symptom_weight[s] if s in matched_symptoms else 0 for s in all_symptoms
    ]

    predicted_disease = model.predict([user_vector])[0]

    med_row = medications[medications["Disease"] == predicted_disease]

    if not med_row.empty:
        # If multiple meds separated by commas, split and strip
        meds_str = med_row["Medication"].values[0]
        medicines = (
            [m.strip() for m in meds_str.split(",")] if "," in meds_str else [meds_str]
        )
    else:
        medicines = ["Consult a doctor"]

    return predicted_disease, medicines


# Get user input
user_input = input("Enter symptoms separated by commas (e.g. itching, fever): ")
user_symptoms = [sym.strip() for sym in user_input.split(",")]

disease, medicines = predict_disease_and_medicine(user_symptoms)

print("Predicted Disease:", disease)

# If medicines is nested list, flatten it
if medicines and isinstance(medicines[0], list):
    medicines = medicines[0]

print("Recommended Medicine:", ", ".join(medicines))
