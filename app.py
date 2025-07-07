import streamlit as st
import joblib
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import re
import numpy as np
from typing import Tuple, Dict, Optional
import json
from datetime import datetime

# Google Generative AI imports
from google import genai
from google.genai import types

st.set_page_config(
    page_title="Enhanced Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

# Configuration
MODEL_NAME = "jy46604790/Fake-News-Bert-Detect"
GEMINI_API_KEY = "AIzaSyB0D21uj-P2PtMQa__UMG2UD4tbi4agngI"  # Replace with your actual API key

# Initialize models
@st.cache_resource
def load_models():
    """Load BERT model and tokenizer"""
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return model, tokenizer, clf

@st.cache_resource
def init_gemini_client():
    """Initialize Gemini client"""
    if not GEMINI_API_KEY:
        return None
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])
    return client, config

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

@st.cache_data
def extract_key_claims(text: str) -> list:
    """Extract key factual claims from text"""
    # Simple claim extraction - you can enhance this with NLP techniques
    sentences = text.split('.')
    claims = []
    
    # Look for sentences with specific patterns that indicate factual claims
    claim_patterns = [
        r'\b(said|stated|announced|reported|confirmed|revealed)\b',
        r'\b(according to|sources|officials|study|research)\b',
        r'\b(\d+%|\d+\s+(people|percent|million|billion|thousand))\b',
        r'\b(happened|occurred|took place|will|was|were)\b'
    ]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20:  # Avoid very short sentences
            for pattern in claim_patterns:
                if re.search(pattern, sentence.lower()):
                    claims.append(sentence)
                    break
    
    return claims[:3]  # Return top 3 claims to avoid API limits

def fact_check_with_ai(text: str, claims: list, client, config) -> Dict:
    """Fact-check claims using Gemini AI"""
    if not client or not claims:
        return {"available": False, "results": []}
    
    try:
        fact_check_results = []
        
        for claim in claims:
            # Create a fact-checking prompt
            prompt = f"""
            Please fact-check this claim and provide a credibility assessment:
            
            Claim: "{claim}"
            
            Please provide:
            1. Verification status (True/False/Partially True/Unverified)
            2. Supporting evidence or sources
            3. Credibility score (0-1 where 1 is most credible)
            4. Brief explanation
            
            Format your response as a structured analysis.
            """
            
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=config,
            )
            
            # Parse the response to extract credibility score
            response_text = response.text
            credibility_score = extract_credibility_score(response_text)
            
            fact_check_results.append({
                "claim": claim,
                "response": response_text,
                "credibility_score": credibility_score
            })
        
        return {"available": True, "results": fact_check_results}
    
    except Exception as e:
        st.error(f"Error in fact-checking: {str(e)}")
        return {"available": False, "results": []}

def extract_credibility_score(response_text: str) -> float:
    """Extract credibility score from AI response"""
    # Look for patterns like "credibility score: 0.8" or "credibility: 0.8"
    patterns = [
        r'credibility\s*score[:\s]*(\d*\.?\d+)',
        r'credibility[:\s]*(\d*\.?\d+)',
        r'score[:\s]*(\d*\.?\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text.lower())
        if match:
            try:
                score = float(match.group(1))
                return min(1.0, max(0.0, score))  # Ensure score is between 0 and 1
            except ValueError:
                continue
    
    # If no explicit score found, try to infer from keywords
    response_lower = response_text.lower()
    if any(word in response_lower for word in ['true', 'verified', 'accurate', 'correct']):
        return 0.8
    elif any(word in response_lower for word in ['false', 'incorrect', 'misleading', 'fake']):
        return 0.2
    elif any(word in response_lower for word in ['partially', 'mixed', 'unclear']):
        return 0.5
    else:
        return 0.5  # Default neutral score

def calculate_combined_score(bert_real: float, bert_fake: float, 
                           fact_check_results: Dict, 
                           bert_weight: float = 0.6, 
                           ai_weight: float = 0.4) -> Dict:
    """Calculate combined confidence score"""
    
    # BERT component
    bert_confidence = max(bert_real, bert_fake)
    bert_prediction = "Real" if bert_real > bert_fake else "Fake"
    
    # AI fact-checking component
    if fact_check_results["available"] and fact_check_results["results"]:
        ai_scores = [result["credibility_score"] for result in fact_check_results["results"]]
        avg_credibility = np.mean(ai_scores)
        ai_confidence = abs(avg_credibility - 0.5) * 2  # Convert to confidence measure
        ai_prediction = "Real" if avg_credibility > 0.5 else "Fake"
    else:
        avg_credibility = 0.5
        ai_confidence = 0.0
        ai_prediction = "Unknown"
    
    # Combined score calculation
    if fact_check_results["available"]:
        # Weighted combination
        if bert_prediction == ai_prediction:
            # Both models agree - boost confidence
            combined_confidence = (bert_weight * bert_confidence + ai_weight * ai_confidence) * 1.2
            combined_prediction = bert_prediction
        else:
            # Models disagree - reduce confidence
            combined_confidence = (bert_weight * bert_confidence + ai_weight * ai_confidence) * 0.8
            # Choose based on higher individual confidence
            if bert_confidence > ai_confidence:
                combined_prediction = bert_prediction
            else:
                combined_prediction = ai_prediction
    else:
        # Only BERT available
        combined_confidence = bert_confidence * 0.8  # Reduce confidence when only one model
        combined_prediction = bert_prediction
    
    combined_confidence = min(1.0, combined_confidence)  # Cap at 1.0
    
    return {
        "prediction": combined_prediction,
        "confidence": combined_confidence,
        "bert_prediction": bert_prediction,
        "bert_confidence": bert_confidence,
        "ai_prediction": ai_prediction,
        "ai_confidence": ai_confidence,
        "ai_credibility": avg_credibility
    }

# Streamlit UI
st.title("📰 Enhanced Fake News Detector")
st.markdown("### AI-Powered News Verification Tool with Fact-Checking")
st.markdown("This application combines BERT classification with AI-powered fact-checking for improved accuracy.")

# Load models
try:
    model, tokenizer, clf = load_models()
except Exception as e:
    st.error(f"❌ Error loading BERT model: {str(e)}")
    st.stop()

# Initialize Gemini client
gemini_client, gemini_config = init_gemini_client()
if gemini_client:
    st.success("Gemini AI fact-checker enabled")
else:
    st.warning("⚠️ Gemini AI not configured - add API key for enhanced fact-checking")

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
    bert_weight = st.slider("BERT Model Weight", 0.1, 0.9, 0.6, 0.1)
    ai_weight = 1.0 - bert_weight
    st.write(f"AI Fact-check Weight: {ai_weight}")
    
    enable_detailed_analysis = st.checkbox("Enable detailed claim analysis", value=True)

if st.button("Analyze News", type="primary"):
    if not content or not content.strip():
        st.warning("Please provide valid content to analyze.")
    else:
        with st.spinner("Analyzing content..."):
            # Clean text
            cleaned = clean_text(content)
            
            # BERT prediction
            bert_real, bert_fake = predict_bert(cleaned, clf)
            
            # Extract claims for fact-checking
            claims = extract_key_claims(content)
            
            # Fact-check with AI
            fact_check_results = fact_check_with_ai(content, claims, gemini_client, gemini_config)
            
            # Calculate combined score
            combined_results = calculate_combined_score(
                bert_real, bert_fake, fact_check_results, bert_weight, ai_weight
            )
        
        # Display results
        st.header("📊 Analysis Results")
        
        # Main prediction
        prediction = combined_results["prediction"]
        confidence = combined_results["confidence"]
        
        if prediction == "Real":
            st.success(f"🟢 **Prediction: {prediction}**")
        else:
            st.error(f"🔴 **Prediction: {prediction}**")
        
        st.metric("Overall Confidence", f"{confidence:.1%}")
        
        # Detailed breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 BERT Model Analysis")
            st.write(f"**Prediction:** {combined_results['bert_prediction']}")
            st.write(f"**Confidence:** {combined_results['bert_confidence']:.1%}")
            st.write(f"**Real Probability:** {bert_real:.3f}")
            st.write(f"**Fake Probability:** {bert_fake:.3f}")
        
        with col2:
            st.subheader("🔍 AI Fact-Check Analysis")
            if fact_check_results["available"]:
                st.write(f"**Prediction:** {combined_results['ai_prediction']}")
                st.write(f"**Confidence:** {combined_results['ai_confidence']:.1%}")
                st.write(f"**Avg Credibility:** {combined_results['ai_credibility']:.3f}")
                st.write(f"**Claims Analyzed:** {len(fact_check_results['results'])}")
            else:
                st.write("**Status:** Not available")
                st.write("*Configure Gemini API key for fact-checking*")
        
        # Detailed claim analysis
        if enable_detailed_analysis and fact_check_results["available"]:
            st.subheader("Detailed Claim Analysis")
            
            for i, result in enumerate(fact_check_results["results"], 1):
                with st.expander(f"Claim {i}: {result['claim'][:100]}..."):
                    st.write(f"**Credibility Score:** {result['credibility_score']:.3f}")
                    st.write("**AI Analysis:**")
                    st.write(result['response'])
        
        # Recommendations
        st.subheader("Recommendations")
        if confidence > 0.8:
            if prediction == "Real":
                st.info("High confidence in authenticity. Content appears credible.")
            else:
                st.warning("High confidence this is fake news. Exercise caution.")
        elif confidence > 0.6:
            st.info("🔍 Moderate confidence. Consider additional verification.")
        else:
            st.warning("Low confidence. Manual fact-checking recommended.")
        
        # Export results
        if st.button("📥 Export Analysis"):
            results_data = {
                "content": content[:200] + "..." if len(content) > 200 else content,
                "timestamp": datetime.now().isoformat(),
                "combined_results": combined_results,
                "bert_scores": {"real": bert_real, "fake": bert_fake},
                "fact_check_available": fact_check_results["available"],
                "claims_analyzed": len(fact_check_results["results"]) if fact_check_results["available"] else 0
            }
            
            st.download_button(
                label="Download Analysis Report",
                data=json.dumps(results_data, indent=2),
                file_name=f"fake_news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Footer
st.markdown("---")
st.markdown("**Note:** This system combines machine learning with AI fact-checking for improved accuracy. Always verify important information through multiple sources.")
