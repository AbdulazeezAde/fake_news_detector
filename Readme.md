# 📰 Fake News Detection System

This project is an AI-powered tool for detecting fake news using BERT-based classification and Gemini AI fact-checking. It provides a Streamlit web interface for users to analyze news articles, headlines, or uploaded text files.

## Features
- BERT-based text classification for fake/real news detection
- Gemini AI-powered fact-checking with web search integration
- Extraction and analysis of key factual claims
- Combined confidence scoring from both models
- Downloadable analysis reports

## How to Use
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set up your Gemini API key:**
   - Create a `.env` file in the project directory:
     ```
     GEMINI_API_KEY = "your-gemini-api-key-here"
     ```
3. **Run the Streamlit app:**
   ```bash
   streamlit run st_app.py
   ```
4. **Interact with the app:**
   - Paste or upload news text, adjust advanced options, and view detailed claim analysis and recommendations.

## File Structure
- `st_app.py` — Main Streamlit application
- `requirements.txt` — Python dependencies
- `.env` — Environment variables (API keys)
- `README.md` — Project documentation

## Notes
- Gemini AI fact-checking requires a valid API key.
- For best results, verify important information through multiple sources.

## License
This project is for educational and research purposes.
