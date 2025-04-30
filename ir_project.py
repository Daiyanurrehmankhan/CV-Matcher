import re
import spacy
import os
import customtkinter as ctk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Hardcoded CV folder path
CV_FOLDER_PATH = r"C:\Users\92321\Desktop\IR_Project\cvs_txt"

def extract_text_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
            return text
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return ""
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', ' ')
    text = text.lower()
    return text

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
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity

def extract_constraints(jd_file):
    jd_text = extract_text_from_file(jd_file)
    if not jd_text:
        return {}

    jd_text = preprocess_text(jd_text)
    doc = nlp(jd_text)

    constraints = {}
    required_skills = []
    skill_keywords = ["must have", "required skills", "requirements", "need", "essential", "should have"]
    for token in doc:
        if token.lower_ in skill_keywords:
            for child in token.children:
                if child.pos_ in ['NOUN', 'PROPN', 'ADJ'] and child.is_alpha:
                    required_skills.append(child.text)
            if token.head.pos_ in ['NOUN', 'PROPN', 'ADJ'] and token.head.is_alpha:
                required_skills.append(token.head.text)

    constraints['required_skills'] = list(set(required_skills))

    experience_keywords = ["experience", "years", "year"]
    for token in doc:
        if token.lower_ in experience_keywords and token.i > 0:
            prev_token = doc[token.i - 1]
            if prev_token.like_num:
                constraints['min_experience'] = float(prev_token.text)
                break

    return constraints

def extract_candidate_preferences(cv_file):
    cv_text = extract_text_from_file(cv_file)
    if not cv_text:
        return {}
    cv_text = preprocess_text(cv_text)
    doc = nlp(cv_text)
    preferences = {}
    preferred_titles = []
    title_keywords = ["title", "position", "role", "job"]
    for token in doc:
        if token.lower_ in title_keywords and token.head.is_alpha:
            preferred_titles.append(token.head.text)
        elif token.pos_ == "JOB_TITLE":
            preferred_titles.append(token.text)
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
    overall_similarity = calculate_similarity(cv_text, jd_text)

    constraints = extract_constraints(jd_file)
    met_constraints = True
    if constraints:
        if 'required_skills' in constraints:
            required_skills = set(constraints['required_skills'])
            candidate_skills = set(cv_entities['skills'])
            if not required_skills.issubset(candidate_skills):
                met_constraints = False
        if 'min_experience' in constraints:
            min_experience = constraints['min_experience']
            years_of_experience = 0
            for exp_item in cv_entities['experience']:
                match = re.search(r'(\d+)\s*(?:years?|yrs?)', exp_item, re.IGNORECASE)
                if match:
                    years_of_experience = max(years_of_experience, int(match.group(1)))
            if years_of_experience < min_experience:
                met_constraints = False

    candidate_fit = 1.0
    if candidate_preferences:
        if 'preferred_titles' in candidate_preferences:
            preferred_titles = [title.lower() for title in candidate_preferences['preferred_titles']]
            job_title_in_jd = False
            for exp_item in jd_entities['experience']:
                if exp_item.lower() in preferred_titles:
                    job_title_in_jd = True
                    break
            if not job_title_in_jd:
                candidate_fit *= 0.5

    final_score = overall_similarity * 0.6 + candidate_fit * 0.2
    if met_constraints:
        final_score += 0.2

    return {
        "overall_similarity": overall_similarity,
        "final_score": final_score,
        "matched_skills": list(set(cv_entities['skills']).intersection(set(jd_entities['skills']))),
        "cv_entities": cv_entities,
        "jd_entities": jd_entities,
        "met_constraints": met_constraints,
        "candidate_fit": candidate_fit,
        "cv_file": cv_file,
    }

def run_matching_ui():
    jd_file_path = jd_file_entry.get()
    if not jd_file_path:
        results_box.delete(1.0, ctk.END)
        results_box.insert(ctk.END, "Please select a job description file.\n")
        return

    jd_filename = os.path.basename(jd_file_path).lower()
    # Extract keyword from JD filename (e.g., "JD-Instructors.txt" → "instructor")
    jd_keyword_match = re.search(r'jd[-_ ]?([a-z]+)', jd_filename)
    jd_keyword = jd_keyword_match.group(1) if jd_keyword_match else None

    if not jd_keyword:
        results_box.delete(1.0, ctk.END)
        results_box.insert(ctk.END, "Could not determine job role from the JD filename.\n")
        return

    match_results = []

    # Process CVs that match the job role keyword
    for filename in os.listdir(CV_FOLDER_PATH):
        if filename.endswith(".txt") and jd_keyword in filename.lower():
            cv_file_path = os.path.join(CV_FOLDER_PATH, filename)
            candidate_preferences = extract_candidate_preferences(cv_file_path)
            match_result = match_cv_to_jd(cv_file_path, jd_file_path, candidate_preferences)
            if "error" not in match_result:
                match_results.append(match_result)

    sorted_results = sorted(match_results, key=lambda x: x['final_score'], reverse=True)

    results_box.delete(1.0, ctk.END)
    if not sorted_results:
        results_box.insert(ctk.END, f"No matching CVs found for role: {jd_keyword}\n")
        return

    for i, result in enumerate(sorted_results[:5]):
        results_box.insert(ctk.END, f"Rank {i + 1}: {os.path.basename(result['cv_file'])}\n")
        results_box.insert(ctk.END, f"  Final Match Score: {result['final_score']:.2f}\n")
        results_box.insert(ctk.END, f"  Overall Similarity: {result['overall_similarity']:.2f}\n\n")

# GUI Setup
app = ctk.CTk()
app.title("CV to Job Description Matcher")
app.geometry("700x600")

# Job Description File Selection
jd_file_label = ctk.CTkLabel(app, text="Select Job Description File:")
jd_file_label.pack(pady=10)

jd_file_entry = ctk.CTkEntry(app, width=400)
jd_file_entry.pack(pady=5)

jd_file_button = ctk.CTkButton(app, text="Browse", command=lambda: jd_file_entry.insert(0, ctk.filedialog.askopenfilename()))
jd_file_button.pack(pady=10)

# Results Display Box
results_box = ctk.CTkTextbox(app, width=600, height=300)
results_box.pack(pady=10)

# Run Button
run_button = ctk.CTkButton(app, text="Run Matching", command=run_matching_ui)
run_button.pack(pady=20)

app.mainloop()