import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/enhanced_dataset.csv")

# Combine symptom columns
symptom_cols = [col for col in df.columns if col.startswith("Symptom")]
df["symptoms"] = df[symptom_cols].values.tolist()
df["symptoms"] = df["symptoms"].apply(
    lambda x: [s.strip().lower() for s in x if pd.notna(s)]
)

# Encode symptoms
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["symptoms"])
y = df["Disease"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model Accuracy:", accuracy)

# Save model and encoder
with open("model/disease_model.pkl", "wb") as f:
    pickle.dump((model, mlb), f)

print("Model saved successfully.")