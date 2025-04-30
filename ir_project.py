import re
import spacy
import os
import customtkinter as ctk
from tkinter import filedialog
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

def extract_text_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

def extract_named_entities(text):
    doc = nlp(text)
    skills, education, experience = [], [], []

    for ent in doc.ents:
        if ent.label_ in ["SKILL", "PROGRAMMING_LANGUAGE"]:
            skills.append(ent.text)
        elif ent.label_ in ["DEGREE", "UNIVERSITY", "SCHOOL"]:
            education.append(ent.text)
        elif ent.label_ in ["WORK_EXPERIENCE", "JOB_TITLE", "ORG"]:
            experience.append(ent.text)

    keywords = ["experience", "skills", "languages", "proficient", "familiar", "knowledge",
                "ability", "expertise", "competent", "capable", "skillset", "tools", "technologies"]
    for token in doc:
        if token.text.lower() in keywords and token.head.is_alpha:
            skills.append(token.head.text)

    return {
        "skills": list(set(skills)),
        "education": list(set(education)),
        "experience": list(set(experience)),
    }

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

def extract_constraints(jd_file):
    jd_text = preprocess_text(extract_text_from_file(jd_file))
    doc = nlp(jd_text)

    constraints = {}
    required_skills = []
    skill_keywords = ["must have", "required skills", "requirements", "need", "essential", "should have"]
    for token in doc:
        if token.lower_ in skill_keywords:
            for child in token.children:
                if child.pos_ in ['NOUN', 'PROPN', 'ADJ']:
                    required_skills.append(child.text)
            if token.head.pos_ in ['NOUN', 'PROPN', 'ADJ']:
                required_skills.append(token.head.text)

    constraints['required_skills'] = list(set(required_skills))

    for token in doc:
        if token.lower_ in ["experience", "years", "year"] and token.i > 0:
            prev = doc[token.i - 1]
            if prev.like_num:
                constraints['min_experience'] = float(prev.text)
                break

    return constraints

def extract_candidate_preferences(cv_file):
    cv_text = preprocess_text(extract_text_from_file(cv_file))
    doc = nlp(cv_text)
    preferences = {}
    preferred_titles = []
    title_keywords = ["title", "position", "role", "job"]
    for token in doc:
        if token.lower_ in title_keywords and token.head.is_alpha:
            preferred_titles.append(token.head.text)
    preferences["preferred_titles"] = list(set(preferred_titles))
    return preferences

def match_cv_to_jd(cv_file, jd_file, candidate_preferences=None):
    cv_text = extract_text_from_file(cv_file)
    jd_text = extract_text_from_file(jd_file)

    if not cv_text or not jd_text:
        return {"error": "Could not read CV or JD file."}

    cv_text = preprocess_text(cv_text)
    jd_text = preprocess_text(jd_text)

    cv_entities = extract_named_entities(cv_text)
    jd_entities = extract_named_entities(jd_text)
    similarity = calculate_similarity(cv_text, jd_text)

    constraints = extract_constraints(jd_file)
    met_constraints = True

    if 'required_skills' in constraints:
        if not set(constraints['required_skills']).issubset(set(cv_entities['skills'])):
            met_constraints = False

    if 'min_experience' in constraints:
        years = 0
        for item in cv_entities['experience']:
            match = re.search(r'(\d+)\s*(?:years?|yrs?)', item, re.IGNORECASE)
            if match:
                years = max(years, int(match.group(1)))
        if years < constraints['min_experience']:
            met_constraints = False

    candidate_fit = 1.0
    if candidate_preferences and 'preferred_titles' in candidate_preferences:
        preferred_titles = [t.lower() for t in candidate_preferences['preferred_titles']]
        title_match = any(exp.lower() in preferred_titles for exp in jd_entities['experience'])
        if not title_match:
            candidate_fit *= 0.5

    final_score = similarity * 0.6 + candidate_fit * 0.2
    if met_constraints:
        final_score += 0.2

    return {
        "overall_similarity": similarity,
        "final_score": final_score,
        "met_constraints": met_constraints,
        "cv_file": cv_file,
    }

def run_matching_ui():
    jd_file = jd_file_entry.get()
    cv_folder = cv_folder_entry.get()
    if not jd_file or not cv_folder:
        results_box.delete(1.0, ctk.END)
        results_box.insert(ctk.END, "Please select both JD file and CV folder.\n")
        return

    jd_keyword = re.search(r'jd[-_ ]?([a-z]+)', os.path.basename(jd_file).lower())
    jd_keyword = jd_keyword.group(1) if jd_keyword else None

    if not jd_keyword:
        results_box.delete(1.0, ctk.END)
        results_box.insert(ctk.END, "Could not determine job role from the JD filename.\n")
        return

    match_results = []
    for filename in os.listdir(cv_folder):
        if filename.endswith(".txt") and jd_keyword in filename.lower():
            cv_path = os.path.join(cv_folder, filename)
            preferences = extract_candidate_preferences(cv_path)
            result = match_cv_to_jd(cv_path, jd_file, preferences)
            if "error" not in result:
                match_results.append(result)

    sorted_results = sorted(match_results, key=lambda x: x['final_score'], reverse=True)
    results_box.delete(1.0, ctk.END)
    if not sorted_results:
        results_box.insert(ctk.END, f"No matching CVs found for role: {jd_keyword}\n")
        return

    for i, r in enumerate(sorted_results[:5]):
        results_box.insert(ctk.END, f"Rank {i+1}: {os.path.basename(r['cv_file'])}\n")
        results_box.insert(ctk.END, f"  Final Match Score: {r['final_score']:.2f}\n")
        results_box.insert(ctk.END, f"  Overall Similarity: {r['overall_similarity']:.2f}\n\n")

app = ctk.CTk()
app.title("CV to JD Matcher")
app.geometry("800x600")
ctk.set_appearance_mode("System")

jd_file_label = ctk.CTkLabel(app, text="Select Job Description File:")
jd_file_label.pack(pady=5)
jd_file_entry = ctk.CTkEntry(app, width=500)
jd_file_entry.pack(pady=5)
jd_browse_btn = ctk.CTkButton(app, text="Browse", command=lambda: jd_file_entry.insert(0, filedialog.askopenfilename()))
jd_browse_btn.pack(pady=5)

cv_folder_label = ctk.CTkLabel(app, text="Select CV Folder:")
cv_folder_label.pack(pady=5)
cv_folder_entry = ctk.CTkEntry(app, width=500)
cv_folder_entry.pack(pady=5)
cv_folder_btn = ctk.CTkButton(app, text="Browse", command=lambda: cv_folder_entry.insert(0, filedialog.askdirectory()))
cv_folder_btn.pack(pady=5)

run_button = ctk.CTkButton(app, text="Run Matching", command=run_matching_ui)
run_button.pack(pady=10)

results_box = ctk.CTkTextbox(app, width=650, height=300)
results_box.pack(pady=10)

app.mainloop()
