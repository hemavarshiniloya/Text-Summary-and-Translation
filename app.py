import streamlit as st
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import pandas as pd
from fpdf import FPDF
from rake_nltk import Rake
import language_tool_python
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator

# Initialize models
model = SentenceTransformer('all-MiniLM-L6-v2')
classifier = pipeline("zero-shot-classification")
translator = Translator()

# Function to summarize text (You can replace with your existing summarizer)
def text_summary(text):
    return text[:min(len(text), 200)]  # Simple placeholder

# Text Similarity Comparison
def compare_texts(text1, text2):
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
    return similarity.item()

# Text Classification
def classify_text(text, candidate_labels):
    return classifier(text, candidate_labels=candidate_labels)

# Keyword Extraction
def extract_keywords(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()

# Grammar and Spell Check
def check_grammar(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    return matches

# Sentiment Analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

# Extract metadata from URL
def extract_metadata(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('title').text if soup.find('title') else "No Title"
        description = soup.find('meta', attrs={'name': 'description'})
        description = description['content'] if description else "No Description"
        return {"title": title, "description": description}
    except Exception as e:
        return {"error": str(e)}

# Save Results as CSV
def save_as_csv(results):
    df = pd.DataFrame(results)
    df.to_csv("summary_results.csv", index=False)

# Save Results as PDF
def save_as_pdf(results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for key, value in results.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    pdf.output("summary_results.pdf")

# Translate text to selected language
def translate_text(text, target_language):
    translation = translator.translate(text, dest=target_language)
    return translation.text

# Streamlit app interface
def main():
    st.title("📝 Enhanced Text Summarization and Translation App")

    # Language selection (For translation)
    selected_language = st.sidebar.selectbox("🌐 Select output language", 
                                             ["en", "es", "fr", "de", "it", "pt", "te", "hi"], 
                                             index=0)

    # Input selection
    input_type = st.selectbox("📂 Select input type", ["Text", "File", "URL"], index=0)

    if input_type == "Text":
        text_input = st.text_area("✏️ Enter text", height=200)

        # Compare and classify button
        if st.button("✨ Summarize, Classify, and Analyze"):
            if text_input:
                with st.spinner("Processing..."):
                    # Summarize, classify, analyze sentiment
                    summary = text_summary(text_input)
                    classification = classify_text(text_input, candidate_labels=["news", "sports", "technology"])
                    sentiment = analyze_sentiment(text_input)
                    keywords = extract_keywords(text_input)
                    grammar_issues = check_grammar(text_input)

                    # Translate the output
                    translated_summary = translate_text(summary, selected_language)
                    translated_classification = translate_text(str(classification), selected_language)
                    translated_sentiment = translate_text(sentiment, selected_language)

                    # Display results
                    st.write("📄 Summary: ", translated_summary)
                    st.write("🗂️ Classification: ", translated_classification)
                    st.write("😊 Sentiment: ", translated_sentiment)
                    st.write("🔑 Keywords: ", keywords)
                    st.write("🔍 Grammar Issues: ", grammar_issues)

                    # Save results
                    save_button = st.button("Save Results")
                    if save_button:
                        save_as_csv({"summary": [translated_summary], "classification": [translated_classification], "sentiment": [translated_sentiment]})
                        save_as_pdf({"summary": translated_summary, "classification": translated_classification, "sentiment": translated_sentiment})

    elif input_type == "URL":
        url_input = st.text_input("🔗 Enter a URL")

        if st.button("✨ Summarize and Analyze URL"):
            if url_input:
                with st.spinner("Processing..."):
                    # Scrape and summarize
                    text = scrape_website(url_input)
                    summary = text_summary(text)
                    metadata = extract_metadata(url_input)

                    # Translate summary
                    translated_summary = translate_text(summary, selected_language)

                    # Display results
                    st.write("📄 URL Summary: ", translated_summary)
                    st.write("📝 URL Metadata: ", metadata)

    elif input_type == "File":
        file_input = st.file_uploader("📂 Upload a file", type=["txt", "pdf", "docx"])

        if file_input is not None:
            # Handle text extraction from file
            text = extract_text_from_file(file_input)  # Define the text extraction function
            summary = text_summary(text)
            translated_summary = translate_text(summary, selected_language)
            st.write("📄 File Summary: ", translated_summary)

# Run the app
if __name__ == "__main__":
    main()
