import pdfplumber
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def extract_pdf_text(path):
    with pdfplumber.open(path) as pdf:
        return " ".join([page.extract_text() or "" for page in pdf.pages])

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    return model.encode(text, convert_to_numpy=True)

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def extract_keywords(text, top_n=10):
    words = [w.lower() for w in text.split() if w.lower() not in ENGLISH_STOP_WORDS and w.isalpha()]
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    return sorted(freq, key=freq.get, reverse=True)[:top_n]
