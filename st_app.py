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
from dotenv import load_dotenv
import os

# Google Generative AI imports
from google import genai
from google.genai import types
from typing import List

load_dotenv()

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

# Configuration
MODEL_NAME = "jy46604790/Fake-News-Bert-Detect"
os.getenv('GEMINI_API_KEY')

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
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key or gemini_key.strip() == "":
        st.error("❌ Gemini API key not found in environment. Please set GEMINI_API_KEY.")
        return None, None

    try:
        client = genai.Client(api_key=gemini_key)
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])
        return client, config
    except Exception as e:
        st.error(f"❌ Failed to initialize Gemini client: {e}")
        return None, None

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
    """Extract key factual claims from text using enhanced NLP techniques"""
    if not text or len(text.strip()) < 10:
        return []
    
    # Clean and prepare text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into sentences using multiple delimiters
    sentences = re.split(r'[.!?]+', text)
    claims = []
    
    # Enhanced claim patterns with more comprehensive coverage
    claim_patterns = [
        # Authority and attribution patterns
        r'\b(said|stated|announced|reported|confirmed|revealed|declared|claimed|alleged|admitted|disclosed)\b',
        r'\b(according to|sources|officials|study|research|investigation|report|analysis|survey)\b',
        r'\b(expert|scientist|researcher|analyst|spokesperson|representative|witness|president|governor|minister)\b',
        
        # Statistical and factual patterns
        r'\b(\d+%|\d+\s*percent|\d+\s+(people|million|billion|thousand|deaths|cases|dollars))\b',
        r'\b(increase|decrease|rise|fall|drop|surge|decline)\s+(\d+|by)',
        r'\b(found|discovered|shows|indicates|proves|demonstrates|reveals)\b',
        
        # Event and temporal patterns
        r'\b(happened|occurred|took place|will|was|were|has been|have been)\b',
        r'\b(today|yesterday|this week|last month|in \d{4}|on \w+day)\b',
        r'\b(breaking|urgent|developing|latest|new|recent|just)\b',
        
        # Cause and effect patterns
        r'\b(caused|led to|resulted in|due to|because of|as a result)\b',
        r'\b(impact|effect|consequence|outcome|result)\b',
        
        # Controversial or newsworthy patterns
        r'\b(scandal|controversy|crisis|emergency|outbreak|attack|arrest|death|fired|resigned)\b',
        r'\b(banned|approved|rejected|suspended|investigated|charged|convicted)\b',
        
        # Comparison and superlative patterns
        r'\b(first|last|only|most|least|highest|lowest|largest|smallest|best|worst)\b',
        r'\b(more than|less than|compared to|versus|against)\b'
    ]
    
    # Additional scoring factors
    importance_keywords = [
        'government', 'president', 'minister', 'official', 'company', 'corporation',
        'health', 'medical', 'vaccine', 'treatment', 'economy', 'market', 'stock',
        'election', 'vote', 'campaign', 'policy', 'law', 'court', 'legal',
        'climate', 'environment', 'technology', 'AI', 'research', 'study', "president"
    ]
    
    sentence_scores = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20 or len(sentence) > 500:  # Filter very short or very long sentences
            continue
            
        score = 0
        matched_patterns = 0
        
        # Score based on claim patterns
        for pattern in claim_patterns:
            if re.search(pattern, sentence.lower()):
                score += 1
                matched_patterns += 1
        
        # Bonus for importance keywords
        for keyword in importance_keywords:
            if keyword in sentence.lower():
                score += 0.5
        
        # Bonus for proper nouns (likely names, places, organizations)
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]+\b', sentence))
        score += proper_nouns * 0.3
        
        # Bonus for numbers and dates
        numbers = len(re.findall(r'\b\d+\b', sentence))
        score += numbers * 0.2
        
        # Penalty for very common words that reduce claim specificity
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        word_count = len(sentence.split())
        common_word_ratio = sum(1 for word in sentence.lower().split() if word in common_words) / word_count
        score -= common_word_ratio
        
        if matched_patterns > 0:  # Only consider sentences with at least one claim pattern
            sentence_scores.append((sentence, score))
    
    # Sort by score and return top claims
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    claims = [sentence for sentence, score in sentence_scores[:5]]  # Increased to 5 for better coverage
    
    return claims

@st.cache_data
def extract_credibility_score(response_text: str) -> float:
    """Extract credibility score from AI response with enhanced pattern matching"""
    if not response_text:
        return 0.5
    
    response_lower = response_text.lower()
    
    # Primary patterns for explicit scores
    score_patterns = [
        r'credibility\s*score[:\s]*(\d*\.?\d+)',
        r'credibility[:\s]*(\d*\.?\d+)',
        r'reliability\s*score[:\s]*(\d*\.?\d+)',
        r'accuracy\s*score[:\s]*(\d*\.?\d+)',
        r'truth\s*score[:\s]*(\d*\.?\d+)',
        r'score[:\s]*(\d*\.?\d+)',
        r'rating[:\s]*(\d*\.?\d+)',
        r'(\d*\.?\d+)\s*\/\s*(?:10|5|1)'  # Fractional scores like 8/10, 0.8/1
    ]
    
    for pattern in score_patterns:
        matches = re.findall(pattern, response_lower)
        if matches:
            try:
                score = float(matches[0])
                # Normalize different scales
                if score > 1.0:
                    if score <= 10:
                        score = score / 10  # Scale from 0-10 to 0-1
                    elif score <= 100:
                        score = score / 100  # Scale from 0-100 to 0-1
                return min(1.0, max(0.0, score))
            except (ValueError, IndexError):
                continue
    
    # Enhanced keyword-based scoring with more nuanced categories
    high_credibility_keywords = [
        'true', 'verified', 'accurate', 'correct', 'factual', 'confirmed', 'legitimate', 'won',
        'authentic', 'reliable', 'trustworthy', 'credible', 'substantiated', 'validated'
    ]
    
    low_credibility_keywords = [
        'false', 'incorrect', 'misleading', 'fake', 'fabricated', 'unverified',
        'debunked', 'disproven', 'inaccurate', 'fraudulent', 'hoax', 'conspiracy'
    ]
    
    moderate_credibility_keywords = [
        'partially', 'mixed', 'unclear', 'uncertain', 'inconclusive', 'disputed',
        'contested', 'questionable', 'unconfirmed', 'alleged', 'claimed'
    ]
    
    # Count keyword occurrences
    high_count = sum(1 for word in high_credibility_keywords if word in response_lower)
    low_count = sum(1 for word in low_credibility_keywords if word in response_lower)
    moderate_count = sum(1 for word in moderate_credibility_keywords if word in response_lower)
    
    # Weight-based scoring
    total_keywords = high_count + low_count + moderate_count
    if total_keywords > 0:
        weighted_score = (high_count * 0.8 + moderate_count * 0.5 + low_count * 0.2) / total_keywords
        return weighted_score
    
    # Context-based scoring for verification status
    if 'verification status' in response_lower:
        if 'true' in response_lower:
            return 0.8
        elif 'false' in response_lower:
            return 0.2
        elif 'partially true' in response_lower:
            return 0.6
        elif 'unverified' in response_lower:
            return 0.4
    
    # Default neutral score
    return 0.5

def fact_check_with_ai(text: str, claims: list, client, config) -> Dict:
    """Enhanced fact-checking with web search integration and improved AI prompts"""
    if not client or not claims:
        return {"available": False, "results": []}
    
    try:
        fact_check_results = []
        
        for i, claim in enumerate(claims):
            st.write(f"Fact-checking claim {i+1}: {claim[:100]}...")
            
            # Enhanced prompt with specific instructions for web search and scoring
            prompt = f"""
            You are a professional fact-checker. Please thoroughly analyze this claim:

            CLAIM: "{claim}"

            Please follow these steps:
            1. First, identify the key factual elements that can be verified
            2. Search for credible sources to verify these elements
            3. Cross-reference multiple sources when possible
            4. Consider the source credibility and recency of information

            Provide your analysis in this format:

            VERIFICATION STATUS: [True/False/Partially True/Unverified/Misleading]

            KEY FINDINGS:
            - [List main findings from your research]

            SOURCES CONSULTED:
            - [List types of sources or specific credible sources if available]

            CREDIBILITY SCORE: [Provide a number between 0.0 and 1.0]
            Where:
            - 0.0-0.2: Clearly false or highly misleading
            - 0.3-0.4: Mostly false with some accurate elements
            - 0.5-0.6: Mixed accuracy or insufficient evidence
            - 0.7-0.8: Mostly true with minor inaccuracies
            - 0.9-1.0: Highly accurate and well-supported

            EXPLANATION:
            [Provide a clear explanation of why you assigned this credibility score]

            CONTEXT:
            [Any important context or nuances that affect the claim's accuracy]

            RED FLAGS (if any):
            [Note any warning signs of misinformation]
            """
            
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                    config=config,
                )
                
                response_text = response.text
                
                # Extract credibility score with enhanced parsing
                credibility_score = extract_credibility_score(response_text)
                
                # Additional analysis for web search keywords
                search_keywords = extract_search_keywords(claim)
                
                # Enhanced result structure
                result = {
                    "claim": claim,
                    "response": response_text,
                    "credibility_score": credibility_score,
                    "search_keywords": search_keywords,
                    "verification_status": extract_verification_status(response_text),
                    "timestamp": datetime.now().isoformat()
                }
                
                fact_check_results.append(result)
                
                # Display results in Streamlit
                st.write(f"**Claim:** {claim}")
                st.write(f"**Credibility Score:** {credibility_score:.2f}")
                with st.expander("Full Analysis"):
                    st.code(response_text)
                st.write("---")
                
            except Exception as e:
                st.error(f"Error processing claim {i+1}: {str(e)}")
                continue
        
        # Calculate overall credibility
        if fact_check_results:
            overall_credibility = sum(r["credibility_score"] for r in fact_check_results) / len(fact_check_results)
            
            return {
                "available": True, 
                "results": fact_check_results,
                "overall_credibility": overall_credibility,
                "total_claims": len(fact_check_results)
            }
        else:
            return {"available": False, "results": []}
            
    except Exception as e:
        st.error(f"Error in fact-checking: {str(e)}")
        return {"available": False, "results": []}

def extract_search_keywords(claim: str) -> List[str]:
    """Extract relevant keywords for web search from a claim"""
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
        'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
        'its', 'our', 'their', 'said', 'says', 'saying'
    }
    
    # Extract important words (proper nouns, numbers, key terms)
    words = re.findall(r'\b[A-Za-z]+\b', claim.lower())
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Also extract proper nouns and numbers from original claim
    proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', claim)
    numbers = re.findall(r'\b\d+\b', claim)
    
    # Combine and deduplicate
    all_keywords = list(set(keywords + [noun.lower() for noun in proper_nouns] + numbers))
    
    return all_keywords[:10]  # Return top 10 keywords

def extract_verification_status(response_text: str) -> str:
    """Extract verification status from AI response"""
    response_lower = response_text.lower()
    
    if 'verification status:' in response_lower:
        # Extract the status after the colon
        status_match = re.search(r'verification status:\s*([^\n\r]+)', response_lower)
        if status_match:
            return status_match.group(1).strip()
    
    # Fallback to keyword detection
    if 'true' in response_lower and 'false' not in response_lower:
        return 'True'
    elif 'false' in response_lower:
        return 'False'
    elif 'partially' in response_lower:
        return 'Partially True'
    elif 'unverified' in response_lower:
        return 'Unverified'
    elif 'misleading' in response_lower:
        return 'Misleading'
    else:
        return 'Unverified'

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

    # Default weights
    bert_weight = 0.2
    ai_weight = 0.8

    # If AI fact check confidence >= 0.45, use AI as default logic and prediction is Real
    if ai_confidence >= 0.45:
        combined_prediction = "Real"
        combined_confidence = ai_confidence * ai_weight + bert_confidence * bert_weight
    else:
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
