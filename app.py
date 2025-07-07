import streamlit as st
import joblib, re
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from google.genai import types



st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)


xgb_pipe = joblib.load('model/baseline_xgb.pkl')
bert_clf = pipeline(
    "text-classification",
    model=AutoModelForSequenceClassification.from_pretrained("jy46604790/Fake-News-Bert-Detect"),
    tokenizer=AutoTokenizer.from_pretrained("jy46604790/Fake-News-Bert-Detect"),
)

# GenAI client
genai = genai.Client()
retrieval_tool = types.Tool(
    google_search_retrieval=types.GoogleSearchRetrieval(
        dynamic_retrieval_config=types.DynamicRetrievalConfig(
            mode=types.DynamicRetrievalConfigMode.MODE_DYNAMIC,
            dynamic_threshold=0.7
        )
    )
)
gen_cfg = types.GenerateContentConfig(tools=[retrieval_tool])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def detect_probs(text):
    p_xgb_real, p_xgb_fake = xgb_pipe.predict_proba([text])[0][1], xgb_pipe.predict_proba([text])[0][0]
    res = bert_clf(text)[0]
    p_bert_real = res['score'] if res['label']=='LABEL_1' else (1-res['score'])
    p_bert_fake = 1 - p_bert_real
    # simple average of detector models
    p_det_real = 0.5*p_xgb_real + 0.5*p_bert_real
    p_det_fake = 1 - p_det_real
    return p_det_real, p_det_fake

def genai_probs(query):
    resp = genai.models.generate_content(
        model='gemini-1.5-flash',
        contents=f"Is this claim real or fake? \"{query}\"",
        config=gen_cfg
    )
    text = resp.text.lower()
    
    if "real" in text and "fake" in text:
        
        import re
        m = re.search(r'(\d+)%', text)
        if m:
            p_fake = int(m.group(1))/100
            p_real = 1 - p_fake
            return p_real, p_fake
    # fallback
    return 0.5, 0.5

def combine(p_det, p_gen, w_det=0.7):
    p_real = w_det*p_det[0] + (1-w_det)*p_gen[0]
    p_fake = 1 - p_real
    return p_real, p_fake


st.title("📰 Fake News Detector")
st.markdown("### AI-Powered News Verification Tool")
st.markdown("This application uses a Fine-tuned BERT model to classify news articles as real or fake.")

inp = st.text_area("Paste headline or text:")
if st.button("Predict"):
    txt = clean_text(inp)
    # 1. Detector
    p_det_real, p_det_fake = detect_probs(txt)
    st.write("**Detector model** — Real:", f"{p_det_real:.2f}", "Fake:", f"{p_det_fake:.2f}")
    # 2. Generative
    p_gen_real, p_gen_fake = genai_probs(txt)
    st.write("**Generative AI** — Real:", f"{p_gen_real:.2f}", "Fake:", f"{p_gen_fake:.2f}")
    # 3. Combined
    p_comb_real, p_comb_fake = combine((p_det_real, p_det_fake), (p_gen_real, p_gen_fake))
    label = "Real News" if p_comb_real>=0.5 else "Fake News"
    st.write("## 🔗 Combined — Prediction:", label)
    st.write(f"Combined Confidence — Real: {p_comb_real:.2f}, Fake: {p_comb_fake:.2f}")
