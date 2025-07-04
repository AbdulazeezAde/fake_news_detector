import streamlit as st
import joblib
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import re

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

# Pretrained Fake News BERT Model
MODEL_NAME = "jy46604790/Fake-News-Bert-Detect"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
clf = pipeline("text-classification", model=model, tokenizer=tokenizer)


@st.cache_data
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@st.cache_data
def predict_xgb(text: str, pipeline):
    # Returns (real_prob, fake_prob)
    if pipeline is None:
        return None, None
    proba = pipeline.predict_proba([text])[0]
    return float(proba[1]), float(proba[0])


@st.cache_data
def predict_bert(text: str):
    
    result = clf(text)[0]
    label = result['label']
    score = result['score']
    
    if label == 'LABEL_1':
        return score, 1 - score
    else:
        return 1 - score, score



st.title("📰 Fake News Detector")
st.markdown("### AI-Powered News Verification Tool")
st.markdown("This application uses a Fine-tuned BERT model to classify news articles as real or fake.")
input_type = st.radio("Select input type:", ['Text',  'File'])

content = ''
if input_type == 'Text':
    content = st.text_area("Enter headline or article text:")
elif input_type == 'File':
    uploaded = st.file_uploader("Upload a text file (.txt)", type=['txt'])
    if uploaded is not None:
        content = uploaded.read().decode('utf-8', errors='ignore')

if st.button("Predict"):
    if not content or not content.strip():
        st.warning("Please provide valid content to classify.")
    else:
        cleaned = clean_text(content)

        # Fake News BERT Model
        bert_real, bert_fake = predict_bert(cleaned)
        label_bert = "Real" if bert_real >= 0.5 else "Fake"
        st.subheader("Fake News Bert Model")
        st.write(f"Prediction: **{label_bert}**")
        st.write(f"Confidence: Real = {bert_real:.2f}, Fake = {bert_fake:.2f}")
