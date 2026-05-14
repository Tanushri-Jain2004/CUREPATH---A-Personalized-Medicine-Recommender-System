from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
import pickle
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.secret_key = "my_secret_key"

# ==============================
# Database Configuration
# ==============================

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "csv"}

# ==============================
# Database Tables
# ==============================

class User(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    name     = db.Column(db.String(150))
    age      = db.Column(db.Integer)
    gender   = db.Column(db.String(20))
    address  = db.Column(db.Text)
    photo    = db.Column(db.String(200))   # profile photo filename


class SearchHistory(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(100))
    symptoms   = db.Column(db.Text)
    prediction = db.Column(db.String(100))
    confidence = db.Column(db.Float)


class ContactMessage(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(150))
    email      = db.Column(db.String(150))
    subject    = db.Column(db.String(200))
    message    = db.Column(db.Text)
    submitted_at = db.Column(db.String(50))


class Consultation(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(100), nullable=False)
    patient_name  = db.Column(db.String(150))
    age           = db.Column(db.Integer)
    gender        = db.Column(db.String(20))
    phone         = db.Column(db.String(30))
    email         = db.Column(db.String(150))
    doctor_pref   = db.Column(db.String(100))   # preferred doctor/speciality
    symptoms      = db.Column(db.Text)
    message       = db.Column(db.Text)
    report_file   = db.Column(db.String(200))   # optional uploaded report filename
    status        = db.Column(db.String(30), default="Pending")  # Pending/Reviewed/Replied
    admin_reply   = db.Column(db.Text)
    submitted_at  = db.Column(db.String(50))


# ==============================
# Load ML model and symptom list
# ==============================

MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_model.pkl")

try:
    model, mlb = pickle.load(open(MODEL_PATH, "rb"))
    symptom_list = list(mlb.classes_)
    print(f"✅ Model loaded. {len(symptom_list)} symptoms found.")
except Exception as e:
    print("❌ Error loading model:", e)
    model, symptom_list = None, []

# ==============================
# Load CSV metadata
# ==============================

def load_data(filename):
    return pd.read_csv(os.path.join(BASE_DIR, "models", filename))

description_df = load_data("description.csv")
description_df.columns = description_df.columns.str.strip()

precautions_df = load_data("precautions.csv")
precautions_df.columns = precautions_df.columns.str.strip()

medications_df = load_data("full_medicine_dataset.csv")
medications_df.columns = medications_df.columns.str.strip()

diets_df = load_data("diet.csv")
diets_df.columns = diets_df.columns.str.strip()

workout_df = load_data("workout.csv")
workout_df.columns = workout_df.columns.str.strip()



# ==============================
# Helper functions
# ==============================

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_prediction(selected_symptoms):
    input_vector = [0] * len(symptom_list)
    for symptom in selected_symptoms:
        if symptom in symptom_list:
            input_vector[symptom_list.index(symptom)] = 1
    prediction    = model.predict([input_vector])[0]
    probabilities = model.predict_proba([input_vector])[0]
    confidence    = round(max(probabilities) * 100, 2)
    return prediction, confidence, probabilities


def get_top_diseases(probabilities, top_n=4):
    """Return top N diseases with their probabilities."""
    classes = model.classes_ if hasattr(model, 'classes_') else []
    if len(classes) == 0:
        return []
    indexed = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)
    results = []
    for idx, prob in indexed[:top_n]:
        if prob > 0.01:  # only include if > 1%
            results.append({
                "disease": classes[idx],
                "confidence": round(prob * 100, 2)
            })
    return results


def find_disease_col(df):
    """Auto-detect the disease column name (handles spacing/casing differences)."""
    for col in df.columns:
        if col.strip().lower() in ("disease", "diseases", "prognosis"):
            return col
    return df.columns[0]  # fallback: first column


def match_disease(df, prediction):
    """
    Match disease name robustly:
    1. Exact match
    2. Case-insensitive match
    3. Strip + lower match
    Returns matched rows DataFrame.
    """
    col = find_disease_col(df)
    # Exact
    rows = df[df[col] == prediction]
    if len(rows): return rows
    # Case-insensitive
    rows = df[df[col].str.lower() == prediction.lower()]
    if len(rows): return rows
    # Strip both sides
    rows = df[df[col].str.strip().str.lower() == prediction.strip().lower()]
    return rows


def clean_list(arr):
    """Remove NaN, None, empty strings from a row slice."""
    return [
        str(x).strip() for x in arr
        if x is not None
        and str(x).strip().lower() not in ("nan", "none", "", "null")
    ]


def get_workout_list(prediction):
    """
    workout_df may be long-format (one row per exercise) OR wide-format.
    Detect and handle both.
    """
    col = find_disease_col(workout_df)
    rows = match_disease(workout_df, prediction)
    if len(rows) == 0:
        return []

    other_cols = [c for c in workout_df.columns if c != col]

    # Wide format: single row, multiple workout columns
    if len(rows) == 1:
        values = clean_list(rows.iloc[0][other_cols].values)
        if values:
            return values

    # Long format: multiple rows, look for a "workout" value column
    for c in other_cols:
        c_lower = c.lower()
        if any(kw in c_lower for kw in ("workout", "exercise", "activity", "recommendation")):
            return clean_list(rows[c].values)

    # Fallback: take first non-disease column
    if other_cols:
        return clean_list(rows[other_cols[0]].values)

    return []


def get_disease_details(prediction):
    # ── Description ──────────────────────────────────────────────────────────
    desc_rows = match_disease(description_df, prediction)
    desc_col  = None
    for c in description_df.columns:
        if c.strip().lower() in ("description", "desc", "details", "about"):
            desc_col = c
            break
    if desc_col is None:
        non_disease = [c for c in description_df.columns
                       if c != find_disease_col(description_df)]
        desc_col = non_disease[0] if non_disease else None

    description = (
        str(desc_rows.iloc[0][desc_col]).strip()
        if len(desc_rows) and desc_col and str(desc_rows.iloc[0][desc_col]).strip().lower() not in ("nan", "none", "")
        else "No description available."
    )

    # ── Precautions ───────────────────────────────────────────────────────────
    prec_rows = match_disease(precautions_df, prediction)
    prec_col  = find_disease_col(precautions_df)
    precautions = (
        clean_list(prec_rows.iloc[0][[c for c in precautions_df.columns if c != prec_col]].values)
        if len(prec_rows) else []
    )

    # ── Medications ───────────────────────────────────────────────────────────
    # ── Medications ───────────────────────────────────────────────────────────
    # ── Medications ───────────────────────────────────────────────────────────
    med_rows = match_disease(medications_df, prediction)

    medications = []
    if len(med_rows):
        med_col = find_disease_col(medications_df)

        # Detect column names automatically
        med_name_col = None
        dose_col = None

        for col in medications_df.columns:
            col_lower = col.lower()
            if "medicine" in col_lower or "drug" in col_lower:
                med_name_col = col
            elif "dose" in col_lower or "mg" in col_lower:
                dose_col = col

        # Combine medicine + dosage
        for _, row in med_rows.iterrows():
            med_name = str(row.get(med_name_col, "")).strip()
            dose = str(row.get(dose_col, "")).strip()

            if med_name and med_name.lower() not in ("nan", "none", ""):
                if dose and dose.lower() not in ("nan", "none", ""):
                    medications.append(f"{med_name} ({dose})")
                else:
                    medications.append(med_name)

    # Remove duplicates
    medications = list(dict.fromkeys(medications))
    # ── Diets ─────────────────────────────────────────────────────────────────
    diet_rows = match_disease(diets_df, prediction)
    diet_col  = find_disease_col(diets_df)
    diets = (
        clean_list(diet_rows.iloc[0][[c for c in diets_df.columns if c != diet_col]].values)
        if len(diet_rows) else []
    )

    # ── Workout ───────────────────────────────────────────────────────────────
    workout = get_workout_list(prediction)

    print(f"[DEBUG] Disease: {prediction!r}")
    print(f"[DEBUG] Description: {description[:60]!r}")
    print(f"[DEBUG] Precautions: {precautions}")
    print(f"[DEBUG] Medications: {medications}")
    print(f"[DEBUG] Diets: {diets}")
    print(f"[DEBUG] Workout: {workout}")

    return {
        "description": description,
        "precautions": precautions,
        "medications": medications,
        "diets":       diets,
        "workout":     workout,
    }


def extract_symptoms_from_message(message):
    message_lower = message.lower().replace("-", " ").replace("_", " ")
    matched = [s for s in symptom_list if s.lower().replace("_", " ") in message_lower]
    return matched


# ==============================
# Conversational AI layer
# ==============================

import re as _re
import random as _random

GREETINGS = [
    "hello", "hi", "hey", "howdy", "hii", "hiii", "yo", "sup",
    "good morning", "good evening", "good afternoon", "good night",
    "what's up", "whats up", "namaste", "greetings"
]

FAREWELLS = [
    "bye", "goodbye", "see you", "take care", "later", "cya",
    "farewell", "good night", "gn", "ttyl"
]

THANKS = [
    "thank you", "thanks", "thankyou", "ty", "thank u",
    "thx", "appreciate", "helpful"
]

HOW_ARE_YOU = [
    "how are you", "how r you", "how are u", "how do you do",
    "are you okay", "you okay", "u ok", "hows it going", "how's it going"
]

HELP_WORDS = [
    "help", "what can you do", "what do you do", "capabilities",
    "features", "how does this work", "how to use", "guide"
]

ABOUT_WORDS = [
    "who are you", "what are you", "tell me about yourself",
    "what is curepath", "about you", "introduce yourself"
]

YES_WORDS = ["yes", "yeah", "yep", "sure", "of course", "definitely",
             "ok", "okay", "alright", "go ahead", "please do", "yup", "ya"]

NO_WORDS  = ["no", "nope", "nah", "not really", "never mind", "skip",
             "no thanks", "don't", "dont"]

GREETING_REPLIES = [
    "Hello {name}! 👋 I'm CurePath, your AI health assistant. How are you feeling today? You can describe your symptoms and I'll help analyse them.",
    "Hi {name}! 😊 Great to see you. I'm CurePath — tell me how you're feeling and I'll help identify possible conditions.",
    "Hey {name}! 👋 I'm CurePath AI. Feel free to describe any symptoms you're experiencing and I'll do my best to help.",
]

FAREWELL_REPLIES = [
    "Take care, {name}! 👋 Remember, your health is your wealth. Come back anytime you need help.",
    "Goodbye {name}! 😊 Stay healthy and don't hesitate to return if you have any concerns.",
    "See you later, {name}! 🌟 Wishing you good health. I'm always here if you need me.",
]

THANKS_REPLIES = [
    "You're welcome, {name}! 😊 I'm always here to help. Is there anything else you'd like to know?",
    "Happy to help, {name}! 🌟 If you have more symptoms or questions, feel free to share.",
    "Glad I could assist, {name}! Take care and stay healthy. 💚",
]

HOW_ARE_YOU_REPLIES = [
    "I'm doing great, thanks for asking {name}! 😊 I'm always ready to help with your health concerns. How are <em>you</em> feeling?",
    "All systems running perfectly! 🤖 More importantly — how are <em>you</em> feeling today, {name}?",
    "I'm wonderful, {name}! Ready to help. Tell me, are you experiencing any symptoms I should know about?",
]

HELP_REPLY = """
<div style="font-family:'DM Sans',sans-serif;">
  <div style="font-size:15px;font-weight:600;color:#04342C;margin-bottom:12px;">Here's what I can do for you 🩺</div>
  <div style="display:flex;flex-direction:column;gap:10px;">
    <div style="background:#E1F5EE;border:1px solid #9FE1CB;border-radius:10px;padding:12px 14px;">
      <div style="font-weight:600;color:#085041;font-size:13px;">🔬 Disease Prediction</div>
      <div style="font-size:13px;color:#5f5e5a;margin-top:3px;">Type your symptoms and I'll predict possible conditions with confidence scores.</div>
    </div>
    <div style="background:#E1F5EE;border:1px solid #9FE1CB;border-radius:10px;padding:12px 14px;">
      <div style="font-weight:600;color:#085041;font-size:13px;">💊 Medications & Diet</div>
      <div style="font-size:13px;color:#5f5e5a;margin-top:3px;">I provide medication suggestions, recommended diets and workouts for each condition.</div>
    </div>
    <div style="background:#E1F5EE;border:1px solid #9FE1CB;border-radius:10px;padding:12px 14px;">
      <div style="font-weight:600;color:#085041;font-size:13px;">🛡️ Precautions</div>
      <div style="font-size:13px;color:#5f5e5a;margin-top:3px;">I list important precautions to follow for your predicted condition.</div>
    </div>
    <div style="background:#E1F5EE;border:1px solid #9FE1CB;border-radius:10px;padding:12px 14px;">
      <div style="font-weight:600;color:#085041;font-size:13px;">📊 Multiple Conditions</div>
      <div style="font-size:13px;color:#5f5e5a;margin-top:3px;">I show other possible conditions ranked by probability.</div>
    </div>
  </div>
  <div style="margin-top:14px;font-size:13.5px;color:#5f5e5a;">
    Just describe how you're feeling — for example: <em style="color:#0F6E56;">I have fever, headache and a dry cough.</em>
  </div>
</div>"""

ABOUT_REPLY = """
<div style="font-family:'DM Sans',sans-serif;">
  <div style="background:linear-gradient(135deg,#E1F5EE,#fff);border:1px solid #9FE1CB;border-radius:12px;padding:16px 18px;">
    <div style="font-size:18px;font-weight:700;color:#04342C;margin-bottom:6px;">👋 I'm CurePath AI</div>
    <div style="font-size:13.5px;color:#5f5e5a;line-height:1.7;">
      I'm an AI-powered health assistant trained to analyse symptoms and predict possible medical conditions.
      I use a machine learning model trained on medical datasets to provide:<br><br>
      ✅ Disease predictions with confidence scores<br>
      ✅ Detailed descriptions of conditions<br>
      ✅ Precautions, medications and diet advice<br>
      ✅ Workout recommendations<br><br>
      <strong>Important:</strong> I'm a decision-support tool — always consult a qualified doctor for proper diagnosis.
    </div>
  </div>
</div>"""


def is_match(message, keywords):
    """
    Exact word-boundary match — prevents 'hi' matching inside 'itching',
    or 'bye' matching inside 'maybe', etc.
    """
    import re as _re2
    msg = message.lower().strip()
    for kw in keywords:
        # Escape the keyword and match as whole word(s)
        pattern = r'(?<![a-z])' + _re2.escape(kw) + r'(?![a-z])'
        if _re2.search(pattern, msg):
            return True
    return False


def is_only_greeting(message, keywords):
    """
    Returns True ONLY if the message is essentially just a greeting
    with no symptom-like content (no medical words).
    """
    msg = message.lower().strip()
    # If message is short (<=4 words) and matches a greeting keyword exactly
    words = msg.split()
    if len(words) <= 4 and is_match(msg, keywords):
        return True
    return False


def get_conversation_state(username):
    """Get current conversation state from session."""
    return session.get(f"conv_{username}", {"step": "idle", "pending": None})


def set_conversation_state(username, state):
    session[f"conv_{username}"] = state
    session.modified = True


def handle_conversation(message, username):
    """
    Stateful conversational handler.
    Returns (reply_html, prediction, handled)
    handled=True means no symptom analysis needed.
    """
    msg   = message.strip()
    name  = username.title()
    state = get_conversation_state(username)

    # ── How are you (check BEFORE greetings — more specific) ─────────────
    if is_match(msg, HOW_ARE_YOU):
        reply = _random.choice(HOW_ARE_YOU_REPLIES).format(name=name)
        return reply, None, True

    # ── Greetings (only if message is short & purely a greeting) ──────────
    if is_only_greeting(msg, GREETINGS):
        set_conversation_state(username, {"step": "greeted", "pending": None})
        reply = _random.choice(GREETING_REPLIES).format(name=name)
        return reply, None, True

    # ── Farewells ─────────────────────────────────────────────────────────
    if is_only_greeting(msg, FAREWELLS):
        set_conversation_state(username, {"step": "idle", "pending": None})
        reply = _random.choice(FAREWELL_REPLIES).format(name=name)
        return reply, None, True

    # ── Thanks ────────────────────────────────────────────────────────────
    if is_only_greeting(msg, THANKS):
        reply = _random.choice(THANKS_REPLIES).format(name=name)
        follow = "<br><br>Would you like to check any other symptoms? Just describe them anytime! 😊"
        return reply + follow, None, True

    # ── Help ──────────────────────────────────────────────────────────────
    if is_match(msg, HELP_WORDS):
        return HELP_REPLY, None, True

    # ── About ─────────────────────────────────────────────────────────────
    if is_match(msg, ABOUT_WORDS):
        return ABOUT_REPLY, None, True

    # ── Yes/No after a pending follow-up ──────────────────────────────────
    if state.get("step") == "asked_more" and is_match(msg, YES_WORDS):
        set_conversation_state(username, {"step": "idle", "pending": None})
        reply = f"Sure {name}! Please describe your symptoms and I'll analyse them right away. 🩺"
        return reply, None, True

    if state.get("step") == "asked_more" and is_match(msg, NO_WORDS):
        set_conversation_state(username, {"step": "idle", "pending": None})
        reply = f"No problem, {name}! Feel free to come back anytime. Take care! 💚"
        return reply, None, True

    # ── "I feel" / "I have" / "I am" natural language ─────────────────────
    # Pass through to symptom extraction — but first set state
    set_conversation_state(username, {"step": "analysing", "pending": None})
    return None, None, False  # not handled — do symptom analysis


def build_list_html(items, color):
    if not items:
        return "<span style='color:#b4b2a9;font-size:13px;'>Not available</span>"
    return "".join(
        f"<li style='margin-bottom:4px;'>{item}</li>" for item in items
    )


def confidence_bar(pct):
    """Returns an inline SVG-style confidence bar."""
    # color based on confidence
    if pct >= 70:
        bar_color = "#1D9E75"
    elif pct >= 40:
        bar_color = "#EF9F27"
    else:
        bar_color = "#E24B4A"

    return f"""
    <div style="display:flex;align-items:center;gap:8px;margin:2px 0 0;">
      <div style="flex:1;height:6px;background:#e8e6df;border-radius:10px;overflow:hidden;">
        <div style="width:{pct}%;height:100%;background:{bar_color};border-radius:10px;transition:width 0.6s;"></div>
      </div>
      <span style="font-size:12px;font-weight:600;color:{bar_color};min-width:42px;">{pct}%</span>
    </div>"""


# ==============================
# Template context processor
# ==============================

@app.context_processor
def inject_current_user():
    if 'user' in session:
        user = User.query.filter_by(username=session['user']).first()
        return {'current_user': user}
    return {'current_user': None}


# ==============================
# Routes
# ==============================

@app.route('/')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("index.html", symptoms=symptom_list)


# ─── Chat endpoint ────────────────────────────────────────────────────────────

@app.route('/chat', methods=['POST'])
def chat():
    if 'user' not in session:
        return jsonify({"reply": "⚠️ Please log in first.", "prediction": None})

    data    = request.get_json()
    message = data.get("message", "").strip()
    username = session['user']

    if not message:
        return jsonify({"reply": "Please describe your symptoms.", "prediction": None})

    # ── Conversational layer first ────────────────────────────────────────
    conv_reply, conv_pred, handled = handle_conversation(message, username)
    if handled:
        return jsonify({"reply": conv_reply, "prediction": conv_pred, "type": "conversation"})

    # ── Symptom extraction ────────────────────────────────────────────────
    matched_symptoms = extract_symptoms_from_message(message)

    if not matched_symptoms:
        name = username.title()
        reply = f"""
        <div style="font-size:14px;line-height:1.8;color:#3c3c3a;">
          Hmm, I couldn't detect any specific symptoms in that, {name}. 🤔<br><br>
          Try describing how you feel more specifically. For example:<br>
          <div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:6px;">
            <span style="background:#E1F5EE;border:1px solid #9FE1CB;color:#085041;font-size:12px;padding:4px 10px;border-radius:20px;">I have fever and chills</span>
            <span style="background:#E1F5EE;border:1px solid #9FE1CB;color:#085041;font-size:12px;padding:4px 10px;border-radius:20px;">dry cough and fatigue</span>
            <span style="background:#E1F5EE;border:1px solid #9FE1CB;color:#085041;font-size:12px;padding:4px 10px;border-radius:20px;">headache and nausea</span>
            <span style="background:#E1F5EE;border:1px solid #9FE1CB;color:#085041;font-size:12px;padding:4px 10px;border-radius:20px;">skin rash and itching</span>
          </div>
          <div style="margin-top:10px;font-size:13px;color:#888780;">
            💡 Tip: Use the autocomplete in the input box to find exact symptom names.
          </div>
        </div>"""
        return jsonify({"reply": reply, "prediction": None, "type": "conversation"})

    # ── Predict ──────────────────────────────────────────────────────────────
    prediction, confidence, probabilities = get_prediction(matched_symptoms)
    details    = get_disease_details(prediction)
    top_diseases = get_top_diseases(probabilities, top_n=4)

    # ── Save to DB ────────────────────────────────────────────────────────────
    db.session.add(SearchHistory(
        username   = session['user'],
        symptoms   = ", ".join(matched_symptoms),
        prediction = prediction,
        confidence = confidence
    ))
    db.session.commit()

    # ── Build concise reply ──────────────────────────────────────────────────
    name = username.title()

    syms_text  = ", ".join(s.replace("_"," ").title() for s in matched_symptoms)
    prec_items = [p for p in details["precautions"] if p and str(p).lower() not in ("nan","none","")][:3]
    med_items  = [m for m in details["medications"]  if m and str(m).lower() not in ("nan","none","")][:2]
    diet_items = [d for d in details["diets"]        if d and str(d).lower() not in ("nan","none","")][:2]
    work_items = [w for w in details["workout"]      if w and str(w).lower() not in ("nan","none","")][:2]

    prec_text  = " • ".join(prec_items) if prec_items else "Consult a doctor."
    med_text   = ", ".join(med_items)   if med_items  else "See a physician."
    diet_text  = ", ".join(diet_items)  if diet_items else "Balanced diet."
    work_text  = ", ".join(work_items)  if work_items else "Light activity."

    conf_color = "#1D9E75" if confidence >= 70 else "#EF9F27" if confidence >= 40 else "#E24B4A"
    desc       = details.get("description") or "No description available."
    desc_short = (desc[:120] + "…") if len(desc) > 120 else desc

    reply = (
        "<div style=\"font-family:'DM Sans',sans-serif;font-size:13.5px;line-height:1.75;\">"
        f'<div style="background:#E1F5EE;border:1px solid #9FE1CB;border-radius:10px;padding:12px 16px;margin-bottom:12px;">'
        f'<span style="font-size:11px;font-weight:700;color:#888780;letter-spacing:0.08em;text-transform:uppercase;">🔬 Prediction</span><br>'
        f'<span style="font-size:18px;font-weight:700;color:#04342C;">{prediction}</span>'
        f'<span style="font-size:12px;font-weight:700;color:{conf_color};margin-left:8px;">({confidence}%)</span>'
        '</div>'
        f'<div style="margin-bottom:7px;"><span style="color:#5f5e5a;font-weight:600;">🩺 Symptoms:</span> {syms_text}</div>'
        f'<div style="margin-bottom:7px;"><span style="color:#5f5e5a;font-weight:600;">📋 About:</span> {desc_short}</div>'
        f'<div style="margin-bottom:7px;"><span style="color:#5f5e5a;font-weight:600;">🛡️ Precautions:</span> {prec_text}</div>'
        f'<div style="margin-bottom:7px;"><span style="color:#5f5e5a;font-weight:600;">💊 Medicines:</span> {med_text}</div>'
        f'<div style="margin-bottom:7px;"><span style="color:#5f5e5a;font-weight:600;">🥗 Diet:</span> {diet_text}</div>'
        f'<div style="margin-bottom:12px;"><span style="color:#5f5e5a;font-weight:600;">🏃 Workout:</span> {work_text}</div>'
        f'<div style="font-size:11.5px;color:#b4b2a9;border-top:1px solid #e8e6df;padding-top:8px;">'
        f'⚕️ AI-based prediction only — always consult a qualified doctor. Any other symptoms to check, {name}?'
        '</div></div>'
    )

    set_conversation_state(username, {"step": "asked_more", "pending": prediction})
    return jsonify({"reply": reply, "prediction": prediction, "type": "prediction"})


# ─── History JSON ─────────────────────────────────────────────────────────────

@app.route('/history_json')
def history_json():
    if 'user' not in session:
        return jsonify({"history": []})
    records = (
        SearchHistory.query
        .filter_by(username=session['user'])
        .order_by(SearchHistory.id.desc())
        .limit(20).all()
    )
    return jsonify({"history": [
        {"symptoms": h.symptoms, "prediction": h.prediction, "confidence": h.confidence}
        for h in records
    ]})




    entry = SearchHistory.query.filter_by(
        id=entry_id, username=session['user']
    ).first()
    if entry:
        db.session.delete(entry)
        db.session.commit()
        return jsonify({"ok": True})
    return jsonify({"ok": False})


# ─── Predict (form-based) ─────────────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist("symptoms")
    if not selected_symptoms:
        return render_template("result.html",
            prediction="No symptoms selected", confidence=0,
            details={"description":"","precautions":[],"medications":[],"diets":[],"workout":[]},
            symptoms=[])

    prediction, confidence, probabilities = get_prediction(selected_symptoms)
    details      = get_disease_details(prediction)
    top_diseases = get_top_diseases(probabilities, top_n=4)

    # Ensure details is never None — fill defaults for missing keys
    if not details:
        details = {}
    details.setdefault("description", "No description available.")
    details.setdefault("precautions", [])
    details.setdefault("medications", [])
    details.setdefault("diets",       [])
    details.setdefault("workout",     [])

    # Log the details for debugging
    print(f"[PREDICT] Disease: {prediction!r}  Conf: {confidence}")
    print(f"[PREDICT] Desc: {str(details.get('description',''))[:60]!r}")
    print(f"[PREDICT] Precautions: {details.get('precautions')}")
    print(f"[PREDICT] Medications: {details.get('medications')}")
    print(f"[PREDICT] Diets:       {details.get('diets')}")
    print(f"[PREDICT] Workout:     {details.get('workout')}")

    if 'user' in session:
        db.session.add(SearchHistory(
            username=session['user'], symptoms=",".join(selected_symptoms),
            prediction=prediction, confidence=confidence))
        db.session.commit()

    return render_template("result.html",
        prediction   = prediction,
        confidence   = confidence,
        details      = details,
        symptoms     = selected_symptoms,
        selected_symptoms = selected_symptoms,
        top_diseases = top_diseases,
        description  = details.get("description", ""),
        precautions  = details.get("precautions", []),
        medications  = details.get("medications", []),
        medicines    = details.get("medications", []),
        diets        = details.get("diets", []),
        diet         = details.get("diets", []),
        workout      = details.get("workout", []),
        others       = [d for d in top_diseases if d["disease"] != prediction]
    )


# ─── Save contact form message ───────────────────────────────────────────────

@app.route('/save_contact', methods=['POST'])
def save_contact():
    from datetime import datetime
    data = request.get_json()
    msg = ContactMessage(
        name    = data.get('name', '').strip(),
        email   = data.get('email', '').strip(),
        subject = data.get('subject', '').strip(),
        message = data.get('message', '').strip(),
        submitted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    db.session.add(msg)
    db.session.commit()
    return jsonify({"ok": True})


# ─── Symptoms list for autocomplete ─────────────────────────────────────────

@app.route('/symptoms_list')
def symptoms_list():
    cleaned = [s.replace("_", " ").strip().title() for s in symptom_list]
    return jsonify({"symptoms": sorted(set(cleaned))})


# ─── Debug route (remove in production) ─────────────────────────────────────

@app.route('/debug_data')
def debug_data():
    """
    Visit /debug_data in your browser to inspect CSV columns and sample rows.
    Remove this route before deploying to production.
    """
    def df_info(name, df):
        col = find_disease_col(df)
        diseases = sorted(df[col].dropna().unique().tolist())
        return {
            "name":     name,
            "columns":  list(df.columns),
            "rows":     len(df),
            "diseases": diseases[:10],   # first 10
            "sample":   df.head(2).to_dict(orient="records"),
        }

    data = [
        df_info("description_df",  description_df),
        df_info("precautions_df",  precautions_df),
        df_info("medications_df",  medications_df),
        df_info("diets_df",        diets_df),
        df_info("workout_df",      workout_df),
    ]

    html = "<html><body style='font-family:monospace;padding:24px;background:#f5f4ef;'><h2>CurePath CSV Debug</h2>"
    for d in data:
        html += f"<hr><h3>{d['name']}</h3>"
        html += f"<b>Rows:</b> {d['rows']} &nbsp; <b>Columns:</b> {d['columns']}<br><br>"
        html += f"<b>First 10 diseases:</b> {d['diseases']}<br><br>"
        html += f"<b>Sample rows:</b><pre>{d['sample']}</pre>"
    html += "</body></html>"
    return html


# ─── Static pages ─────────────────────────────────────────────────────────────

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route("/guideline")
def guideline():
    return render_template("guideline.html")


# ─── Signup ───────────────────────────────────────────────────────────────────

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name     = request.form['name']
        age      = request.form['age']
        gender   = request.form['gender']
        address  = request.form['address']
        if User.query.filter_by(username=username).first():
            return render_template("signup.html", error="Username already taken.")
        db.session.add(User(username=username, password=password,
            name=name, age=age, gender=gender, address=address))
        db.session.commit()
        return redirect(url_for('login'))
    return render_template("signup.html")


# ─── Login ────────────────────────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user'] = username
            return redirect(url_for('home'))
        return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html")


# ─── History page ─────────────────────────────────────────────────────────────

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))
    user_history = SearchHistory.query.filter_by(username=session['user']).all()
    return render_template("history.html", history=user_history)


# ─── Upload report ────────────────────────────────────────────────────────────

@app.route("/upload_report", methods=["GET", "POST"])
def upload_report():
    if request.method == "POST":
        file = request.files["report"]
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            return render_template("report_result.html",
                prediction="Possible Anemia", confidence=82.5, file=file.filename)
    return render_template("upload_report.html")


# ─── Book Consultation ────────────────────────────────────────────────────────

CONSULTATION_UPLOAD = os.path.join(BASE_DIR, 'static', 'consultation_reports')

@app.route('/book_consultation', methods=['GET', 'POST'])
def book_consultation():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = User.query.filter_by(username=session['user']).first()
    success = False

    if request.method == 'POST':
        from datetime import datetime
        os.makedirs(CONSULTATION_UPLOAD, exist_ok=True)

        report_filename = None
        report_file = request.files.get('report')
        if report_file and report_file.filename:
            ext = report_file.filename.rsplit('.', 1)[-1].lower()
            if ext in ('pdf', 'png', 'jpg', 'jpeg'):
                import uuid
                report_filename = f"{session['user']}_{uuid.uuid4().hex[:8]}.{ext}"
                report_file.save(os.path.join(CONSULTATION_UPLOAD, report_filename))

        consult = Consultation(
            username     = session['user'],
            patient_name = request.form.get('patient_name', '').strip(),
            age          = request.form.get('age') or None,
            gender       = request.form.get('gender', '').strip(),
            phone        = request.form.get('phone', '').strip(),
            email        = request.form.get('email', '').strip(),
            doctor_pref  = request.form.get('doctor_pref', '').strip(),
            symptoms     = request.form.get('symptoms', '').strip(),
            message      = request.form.get('message', '').strip(),
            report_file  = report_filename,
            status       = 'Pending',
            submitted_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        )
        db.session.add(consult)
        db.session.commit()
        success = True

    my_consultations = Consultation.query.filter_by(username=session['user'])                       .order_by(Consultation.id.desc()).all()
    return render_template("book_consultation.html",
        user=user, success=success,
        my_consultations=my_consultations)


# ─── Profile ──────────────────────────────────────────────────────────────────

@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['user']).first_or_404()
    total_searches = SearchHistory.query.filter_by(username=session['user']).count()
    return render_template("profile.html", user=user, total_searches=total_searches)


@app.route('/profile/edit', methods=['GET', 'POST'])
def profile_edit():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['user']).first_or_404()

    if request.method == 'POST':
        user.name    = request.form.get('name', '').strip()
        user.age     = request.form.get('age', None) or None
        user.gender  = request.form.get('gender', '').strip()
        user.address = request.form.get('address', '').strip()

        # Password change (optional)
        new_pass = request.form.get('new_password', '').strip()
        if new_pass:
            user.password = new_pass

        db.session.commit()
        return redirect(url_for('profile'))

    return render_template("profile_edit.html", user=user)


@app.route('/profile/upload_photo', methods=['POST'])
def upload_profile_photo():
    if 'user' not in session:
        return jsonify({"ok": False, "error": "Not logged in"})

    if 'photo' not in request.files:
        return jsonify({"ok": False, "error": "No file"})

    file = request.files['photo']
    if file.filename == '':
        return jsonify({"ok": False, "error": "No file selected"})

    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in ('png', 'jpg', 'jpeg', 'gif', 'webp'):
        return jsonify({"ok": False, "error": "Invalid file type"})

    import uuid
    photo_dir = os.path.join(BASE_DIR, 'static', 'profile_photos')
    os.makedirs(photo_dir, exist_ok=True)

    filename  = f"{session['user']}_{uuid.uuid4().hex[:8]}.{ext}"
    filepath  = os.path.join(photo_dir, filename)

    # Remove old photo if exists
    user = User.query.filter_by(username=session['user']).first()
    if user and user.photo:
        old_path = os.path.join(photo_dir, user.photo)
        if os.path.exists(old_path):
            os.remove(old_path)

    file.save(filepath)
    user.photo = filename
    db.session.commit()

    return jsonify({"ok": True, "filename": filename})


# ─── Logout ───────────────────────────────────────────────────────────────────

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('admin', None)
    return redirect(url_for('login'))


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN PANEL
# ══════════════════════════════════════════════════════════════════════════════

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "curepath@admin123"


def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    error = None
    if request.method == 'POST':
        if (request.form.get('username') == ADMIN_USERNAME and
                request.form.get('password') == ADMIN_PASSWORD):
            session['admin'] = True
            return redirect(url_for('admin_dashboard'))
        error = "Invalid admin credentials."
    return render_template("admin_login.html", error=error)


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    return redirect(url_for('admin_login'))


@app.route('/admin')
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    total_users    = User.query.count()
    total_searches = SearchHistory.query.count()
    total_contacts = ContactMessage.query.count()
    recent_searches = (SearchHistory.query
                       .order_by(SearchHistory.id.desc())
                       .limit(5).all())
    recent_contacts = (ContactMessage.query
                       .order_by(ContactMessage.id.desc())
                       .limit(5).all())
    # Top predicted diseases
    from sqlalchemy import func
    top_diseases = (db.session.query(
        SearchHistory.prediction,
        func.count(SearchHistory.id).label('count'))
        .group_by(SearchHistory.prediction)
        .order_by(func.count(SearchHistory.id).desc())
        .limit(5).all())

    total_consultations = Consultation.query.count()
    return render_template("admin_dashboard.html",
        total_users=total_users,
        total_searches=total_searches,
        total_contacts=total_contacts,
        total_consultations=total_consultations,
        recent_searches=recent_searches,
        recent_contacts=recent_contacts,
        top_diseases=top_diseases)


@app.route('/admin/users')
@admin_required
def admin_users():
    q = request.args.get('q', '').strip()
    if q:
        users = User.query.filter(
            (User.username.ilike(f'%{q}%')) |
            (User.name.ilike(f'%{q}%')) |
            (User.email.ilike(f'%{q}%') if hasattr(User, 'email') else False)
        ).all()
    else:
        users = User.query.order_by(User.id.desc()).all()
    # Attach search count per user
    from sqlalchemy import func
    counts = dict(db.session.query(
        SearchHistory.username,
        func.count(SearchHistory.id))
        .group_by(SearchHistory.username).all())
    return render_template("admin_users.html", users=users, counts=counts, q=q)


@app.route('/admin/users/delete/<int:uid>', methods=['POST'])
@admin_required
def admin_delete_user(uid):
    user = User.query.get_or_404(uid)
    SearchHistory.query.filter_by(username=user.username).delete()
    db.session.delete(user)
    db.session.commit()
    return redirect(url_for('admin_users'))


@app.route('/admin/searches')
@admin_required
def admin_searches():
    q    = request.args.get('q', '').strip()
    user = request.args.get('user', '').strip()
    if q:
        searches = SearchHistory.query.filter(
            (SearchHistory.prediction.ilike(f'%{q}%')) |
            (SearchHistory.symptoms.ilike(f'%{q}%')) |
            (SearchHistory.username.ilike(f'%{q}%'))
        ).order_by(SearchHistory.id.desc()).all()
    elif user:
        searches = SearchHistory.query.filter_by(username=user)                   .order_by(SearchHistory.id.desc()).all()
    else:
        searches = SearchHistory.query.order_by(SearchHistory.id.desc()).all()
    return render_template("admin_searches.html", searches=searches, q=q, user=user)


@app.route('/admin/searches/delete/<int:sid>', methods=['POST'])
@admin_required
def admin_delete_search(sid):
    entry = SearchHistory.query.get_or_404(sid)
    db.session.delete(entry)
    db.session.commit()
    return redirect(url_for('admin_searches'))


@app.route('/admin/contacts')
@admin_required
def admin_contacts():
    q = request.args.get('q', '').strip()
    if q:
        contacts = ContactMessage.query.filter(
            (ContactMessage.name.ilike(f'%{q}%')) |
            (ContactMessage.email.ilike(f'%{q}%')) |
            (ContactMessage.subject.ilike(f'%{q}%'))
        ).order_by(ContactMessage.id.desc()).all()
    else:
        contacts = ContactMessage.query.order_by(ContactMessage.id.desc()).all()
    return render_template("admin_contacts.html", contacts=contacts, q=q)


@app.route('/admin/contacts/delete/<int:cid>', methods=['POST'])
@admin_required
def admin_delete_contact(cid):
    msg = ContactMessage.query.get_or_404(cid)
    db.session.delete(msg)
    db.session.commit()
    return redirect(url_for('admin_contacts'))


@app.route('/admin/contacts/view/<int:cid>')
@admin_required
def admin_view_contact(cid):
    msg = ContactMessage.query.get_or_404(cid)
    return jsonify({
        "name": msg.name, "email": msg.email,
        "subject": msg.subject, "message": msg.message,
        "submitted_at": msg.submitted_at
    })


@app.route('/admin/user_searches/<username>')
@admin_required
def admin_user_searches(username):
    user     = User.query.filter_by(username=username).first_or_404()
    searches = SearchHistory.query.filter_by(username=username)               .order_by(SearchHistory.id.desc()).all()
    return render_template("admin_user_searches.html", user=user, searches=searches)


@app.route('/admin/consultations')
@admin_required
def admin_consultations():
    q = request.args.get('q', '').strip()
    status_filter = request.args.get('status', '').strip()
    query = Consultation.query
    if q:
        query = query.filter(
            (Consultation.patient_name.ilike(f'%{q}%')) |
            (Consultation.username.ilike(f'%{q}%')) |
            (Consultation.doctor_pref.ilike(f'%{q}%'))
        )
    if status_filter:
        query = query.filter_by(status=status_filter)
    consultations = query.order_by(Consultation.id.desc()).all()
    return render_template("admin_consultations.html",
        consultations=consultations, q=q, status_filter=status_filter)


@app.route('/admin/consultations/reply/<int:cid>', methods=['POST'])
@admin_required
def admin_reply_consultation(cid):
    consult = Consultation.query.get_or_404(cid)
    consult.admin_reply = request.form.get('reply', '').strip()
    consult.status      = request.form.get('status', 'Reviewed')
    db.session.commit()
    return redirect(url_for('admin_consultations'))


@app.route('/admin/consultations/delete/<int:cid>', methods=['POST'])
@admin_required
def admin_delete_consultation(cid):
    consult = Consultation.query.get_or_404(cid)
    if consult.report_file:
        path = os.path.join(CONSULTATION_UPLOAD, consult.report_file)
        if os.path.exists(path):
            os.remove(path)
    db.session.delete(consult)
    db.session.commit()
    return redirect(url_for('admin_consultations'))


@app.route('/admin/user_profile/<username>')
@admin_required
def admin_user_profile(username):
    user = User.query.filter_by(username=username).first_or_404()
    searches = SearchHistory.query.filter_by(username=username)               .order_by(SearchHistory.id.desc()).all()
    consultations = Consultation.query.filter_by(username=username)                   .order_by(Consultation.id.desc()).all()
    return render_template("admin_user_profile.html",
        user=user, searches=searches, consultations=consultations)


# ==============================
# Create Database & Run
# ==============================

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)