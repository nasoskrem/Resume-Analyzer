# 🧠 Resume Analyzer (with AI & PyTorch)

A smart web app that compares your resume against a job description using deep semantic similarity and keyword analysis. Built with Flask, PyTorch (Sentence-BERT), and modern HTML/CSS.

---

## 🚀 Features

- 🔍 **AI-Powered Resume Matching**  
  Uses `sentence-transformers` (BERT-based) to compute semantic similarity between your resume and the job description.

- 📊 **Match Score**  
  Shows a score (0–100%) reflecting how well your resume aligns with the job posting.

- 🧠 **Missing Keyword Extraction**  
  Highlights relevant job keywords missing from your resume.

- 💡 **Modern Web UI**  
  Clean, responsive interface built with HTML5 + CSS3.

---

## 📦 Installation

```bash
git clone https://github.com/nasoskrem/Resume-Analyzer.git
cd Resume-Analyzer
pip install -r requirements.txt
python app.py
