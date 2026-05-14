import pandas as pd
import pickle
import numpy as np

# ================= LOAD TRAINED MODEL ================= #
with open("model/disease_model.pkl", "rb") as f:
    model, mlb = pickle.load(f)

# ================= LOAD DATASETS ================= #
medicine_df = pd.read_csv("data/full_medicine_dataset.csv")
diet_df = pd.read_csv("data/diet.csv")
workout_df = pd.read_csv("data/workout.csv")
description_df = pd.read_csv("data/description.csv")
precautions_df = pd.read_csv("data/precautions.csv")
disease_df = pd.read_csv("data/enhanced_dataset.csv")

# ================= NORMALIZATION ================= #
def norm(text):
    return str(text).strip().lower()

for df in [medicine_df, diet_df, workout_df, description_df, precautions_df, disease_df]:
    df["Disease"] = df["Disease"].apply(norm)

symptom_cols = [col for col in disease_df.columns if col.startswith("Symptom")]

# ================= PRIMARY ML PREDICTION ================= #
def predict_primary_disease(symptoms):
    vec = mlb.transform([symptoms])
    probs = model.predict_proba(vec)[0]
    idx = np.argmax(probs)
    return model.classes_[idx], probs[idx]

# ================= SECONDARY DISEASES (DEDUPLICATED) ================= #
def get_other_diseases(user_symptoms, primary_disease, top_k=2):
    disease_scores = {}

    for _, row in disease_df.iterrows():
        disease = row["Disease"]
        if disease == primary_disease:
            continue

        disease_symptoms = {
            norm(row[col]) for col in symptom_cols if pd.notna(row[col])
        }

        common = disease_symptoms & user_symptoms
        if not common:
            continue

        similarity = len(common) / len(disease_symptoms)

        if (
            disease not in disease_scores or
            similarity > disease_scores[disease]["similarity"]
        ):
            disease_scores[disease] = {
                "similarity": similarity,
                "overlap": len(common)
            }

    ranked = [
        {"disease": d, **v}
        for d, v in disease_scores.items()
    ]

    ranked.sort(
        key=lambda x: (x["similarity"], x["overlap"]),
        reverse=True
    )

    return ranked[:top_k]

# ================= FETCH RECOMMENDATIONS ================= #
def get_recommendations(disease):
    return {
        "desc": description_df[description_df["Disease"] == disease],
        "med": medicine_df[medicine_df["Disease"] == disease],
        "diet": diet_df[diet_df["Disease"] == disease],
        "work": workout_df[workout_df["Disease"] == disease],
        "prec": precautions_df[precautions_df["Disease"] == disease]
    }

# ================= DISPLAY ================= #
def display_results(disease, confidence, data):
    print("\n" + "=" * 70)
    print(f"Disease (based on your symptoms): {disease.title()}")
    print(f"Confidence Level                : {round(confidence * 100, 2)}%")
    print("=" * 70)

    if not data["desc"].empty:
        print("\nDescription:")
        print(f"- {data['desc']['Description'].values[0]}")

    print("\nRecommended Medicines:")
    if not data["med"].empty:
        for _, r in data["med"].iterrows():
            print(f"- {r['Medicine']} ({r['Dosage_mg']} mg)")
    else:
        print("- Consult a medical professional")

    print("\nDiet Plan:")
    if not data["diet"].empty:
        row = data["diet"].iloc[0]
        for i in range(1, 6):
            if pd.notna(row.get(f"Diet_{i}")):
                print(f"- {row[f'Diet_{i}']}")
    else:
        print("- Balanced diet recommended")

    print("\nWorkout / Activity:")
    if not data["work"].empty:
        row = data["work"].iloc[0]
        for i in range(1, 5):
            if pd.notna(row.get(f"Workout_{i}")):
                print(f"- {row[f'Workout_{i}']}")
    else:
        print("- Light physical activity recommended")

    print("\nPrecautions:")
    if not data["prec"].empty:
        row = data["prec"].iloc[0]
        for col in row.index:
            if col != "Disease" and pd.notna(row[col]):
                print(f"- {row[col]}")

    print("\n⚠ Medical Disclaimer:")
    print("This system is for educational purposes only.")
    print("=" * 70)

# ================= MAIN ================= #
if __name__ == "__main__":

    user_input = input("Enter symptoms (comma separated): ")
    user_symptoms = {norm(s) for s in user_input.split(",") if s.strip()}

    disease, confidence = predict_primary_disease(list(user_symptoms))
    disease = norm(disease)

    display_results(disease, confidence, get_recommendations(disease))

    others = get_other_diseases(user_symptoms, disease)

    if others:
        print("\nOther Possible Diseases (Symptom Similarity):")
        for o in others:
            print(
                f"- {o['disease'].title()} "
                f"(Similarity: {round(o['similarity'] * 100, 2)}%, "
                f"Common Symptoms: {o['overlap']})"
            )