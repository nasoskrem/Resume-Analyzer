from flask import Flask, request, render_template
import os
from utils.text_utils import extract_pdf_text, get_embedding, cosine_similarity, extract_keywords
from utils.file_utils import save_uploaded_file

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    match_score = None
    missing_keywords = []

    if request.method == "POST":
        resume_file = request.files["resume"]
        job_text = request.form["job_desc"]

        if resume_file and job_text:
            resume_path = save_uploaded_file(resume_file, app.config["UPLOAD_FOLDER"])
            resume_text = extract_pdf_text(resume_path)
            resume_vec = get_embedding(resume_text)
            job_vec = get_embedding(job_text)

            score = cosine_similarity(resume_vec, job_vec)
            match_score = round(max(0, min(score, 1.0)) * 100, 2)

            resume_keywords = set(extract_keywords(resume_text))
            job_keywords = set(extract_keywords(job_text))
            missing_keywords = list(job_keywords - resume_keywords)

    return render_template("index.html", score=match_score, missing=missing_keywords)

if __name__ == "__main__":
    app.run(debug=True)
