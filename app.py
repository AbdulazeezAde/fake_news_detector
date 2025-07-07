import streamlit as st
import joblib
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import re
import numpy as np
from typing import Tuple
import json
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

# Configuration
MODEL_NAME = "jy46604790/Fake-News-Bert-Detect"

# Initialize models
@st.cache_resource
def load_models():
    """Load BERT model and tokenizer"""
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return model, tokenizer, clf

# Text processing functions
@st.cache_data
def clean_text(text: str) -> str:
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_data
def predict_bert(text: str, _clf) -> Tuple[float, float]:
    """Predict using BERT model"""
    result = _clf(text)[0]
    label = result['label']
    score = result['score']

    if label == 'LABEL_1':
        return score, 1 - score  # real_prob, fake_prob
    else:
        return 1 - score, score

def get_prediction_category(real_prob: float, fake_prob: float) -> str:
    """Get prediction category based on probabilities"""
    confidence = max(real_prob, fake_prob)
    
    if confidence > 0.8:
        return "High Confidence"
    elif confidence > 0.6:
        return "Moderate Confidence"
    else:
        return "Low Confidence"

def get_recommendation(prediction: str, confidence: float) -> str:
    """Get recommendation based on prediction and confidence"""
    if confidence > 0.8:
        if prediction == "Real":
            return "✅ High confidence in authenticity. Content appears credible."
        else:
            return "⚠️ High confidence this is fake news. Exercise caution."
    elif confidence > 0.6:
        return "🔍 Moderate confidence. Consider additional verification from multiple sources."
    else:
        return "❓ Low confidence. Manual fact-checking strongly recommended."

# Streamlit UI
st.title("📰 Fake News Detection System")
st.markdown("### BERT-Powered News Verification Tool")
st.markdown("This application uses a fine-tuned BERT model to classify news articles as real or fake.")

# Load models
try:
    model, tokenizer, clf = load_models()
    st.success("✅ BERT model loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading BERT model: {str(e)}")
    st.stop()

# User input
input_type = st.radio("Select input type:", ['Text', 'File'])

content = ''
if input_type == 'Text':
    content = st.text_area("Enter headline or article text:", height=150)
elif input_type == 'File':
    uploaded = st.file_uploader("Upload a text file (.txt)", type=['txt'])
    if uploaded is not None:
        content = uploaded.read().decode('utf-8', errors='ignore')

# Advanced options
with st.expander("Advanced Options"):
    show_raw_scores = st.checkbox("Show raw prediction scores", value=False)
    show_cleaned_text = st.checkbox("Show cleaned text", value=False)

if st.button("Analyze News", type="primary"):
    if not content or not content.strip():
        st.warning("Please provide valid content to analyze.")
    else:
        with st.spinner("Analyzing content..."):
            # Clean text
            cleaned = clean_text(content)

            # BERT prediction
            real_prob, fake_prob = predict_bert(cleaned, clf)
            
            # Determine prediction
            prediction = "Real" if real_prob > fake_prob else "Fake"
            confidence = max(real_prob, fake_prob)
            confidence_category = get_prediction_category(real_prob, fake_prob)

        # Display results
        st.header("📊 Analysis Results")

        # Main prediction
        if prediction == "Real":
            st.success(f"🟢 **Prediction: {prediction}**")
        else:
            st.error(f"🔴 **Prediction: {prediction}**")

        # Confidence metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Confidence", f"{confidence:.1%}")
        
        with col2:
            st.metric("Confidence Level", confidence_category)

        # Detailed breakdown
        st.subheader("🤖 BERT Model Analysis")
        
        # Probability bars
        st.write("**Probability Breakdown:**")
        st.progress(real_prob, text=f"Real: {real_prob:.1%}")
        st.progress(fake_prob, text=f"Fake: {fake_prob:.1%}")

        # Raw scores if requested
        if show_raw_scores:
            st.subheader("📈 Raw Prediction Scores")
            st.write(f"**Real Probability:** {real_prob:.6f}")
            st.write(f"**Fake Probability:** {fake_prob:.6f}")
            st.write(f"**Confidence Score:** {confidence:.6f}")

        # Cleaned text if requested
        if show_cleaned_text:
            st.subheader("🧹 Cleaned Text")
            st.text_area("Processed text used for analysis:", cleaned, height=100)

        # Recommendations
        st.subheader("💡 Recommendations")
        recommendation = get_recommendation(prediction, confidence)
        
        if confidence > 0.8:
            if prediction == "Real":
                st.info(recommendation)
            else:
                st.warning(recommendation)
        elif confidence > 0.6:
            st.info(recommendation)
        else:
            st.warning(recommendation)

        # Additional context
        st.subheader("📋 Additional Context")
        st.write("**Text Length:** {} characters".format(len(content)))
        st.write("**Cleaned Text Length:** {} characters".format(len(cleaned)))
        st.write("**Model:** {}".format(MODEL_NAME))
        st.write("**Analysis Time:** {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        # Export results
        if st.button("📥 Export Analysis"):
            results_data = {
                "content": content[:200] + "..." if len(content) > 200 else content,
                "timestamp": datetime.now().isoformat(),
                "prediction": prediction,
                "confidence": confidence,
                "confidence_category": confidence_category,
                "real_probability": real_prob,
                "fake_probability": fake_prob,
                "text_length": len(content),
                "cleaned_text_length": len(cleaned),
                "model_name": MODEL_NAME,
                "recommendation": recommendation
            }

            st.download_button(
                label="Download Analysis Report",
                data=json.dumps(results_data, indent=2),
                file_name=f"fake_news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Information section
st.markdown("---")
st.subheader("ℹ️ About This Tool")
st.markdown("""
This fake news detection system uses a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model 
to classify news articles as real or fake. The model has been trained on a large dataset of news articles and can 
identify patterns that distinguish authentic journalism from misinformation.

**How it works:**
1. **Text Preprocessing:** The input text is cleaned and normalized
2. **BERT Analysis:** The pre-trained model analyzes the text patterns
3. **Classification:** The model outputs probabilities for real vs fake news
4. **Confidence Assessment:** The system provides confidence levels for the prediction

**Limitations:**
- The model's performance depends on the quality of its training data
- It may not perform well on very recent events or emerging topics
- Always verify important information through multiple credible sources
- This tool should be used as a starting point for fact-checking, not as definitive proof
""")

# Footer
st.markdown("---")
st.markdown("**Note:** This system uses machine learning for news verification. Always verify important information through multiple credible sources.")
st.markdown("**Model:** jy46604790/Fake-News-Bert-Detect from Hugging Face")
