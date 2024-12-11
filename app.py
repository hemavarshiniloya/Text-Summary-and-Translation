import nltk
from rake_nltk import Rake
import streamlit as st
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import pandas as pd
from fpdf import FPDF
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator, LANGUAGES
from PyPDF2 import PdfReader
from docx import Document
import language_tool_python

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')  # This downloads the missing punkt_tab resource
nltk.download('stopwords')

# Initialize models
model = SentenceTransformer('all-MiniLM-L6-v2')
classifier = pipeline("zero-shot-classification")

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
def translate_text(text, target_language="en"):
    translator = Translator()
    return translator.translate(text, dest=target_language).text

# Function to extract text from files (PDF, DOCX, TXT)
def extract_text_from_file(file):
    if file.type == "text/plain":
        return file.getvalue().decode("utf-8")
    
    elif file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    
    return ""  # Return empty string if file type is not supported

# Scrape website to extract text
def scrape_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = " ".join([para.text for para in paragraphs])
        return text
    except Exception as e:
        return f"Error scraping the website: {e}"

# Streamlit app interface
def main():
    st.title("📝 Enhanced Text Summarization and Translation App")

    # Language selection (All available languages)
    language_list = list(LANGUAGES.keys())  # List of all available language codes
    selected_language_code = st.sidebar.selectbox("🌐 Select language", language_list, index=0)
    selected_language_name = LANGUAGES[selected_language_code]  # Get the language name

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

                    # Display results
                    st.write("📄 Summary: ", summary)
                    st.write("🗂️ Classification: ", classification)
                    st.write("😊 Sentiment: ", sentiment)
                    st.write("🔑 Keywords: ", keywords)
                    st.write("🔍 Grammar Issues: ", grammar_issues)

                    # Translate if not in English
                    if selected_language_code != "en":
                        summary = translate_text(summary, target_language=selected_language_code)
                        # Apply translation to other results as needed
                        classification = translate_text(str(classification), target_language=selected_language_code)
                        sentiment = translate_text(str(sentiment), target_language=selected_language_code)

                    # Display translated results
                    st.write(f"🗣️ Translated Summary in {selected_language_name}: ", summary)
                    st.write(f"🗣️ Translated Classification in {selected_language_name}: ", classification)
                    st.write(f"🗣️ Translated Sentiment in {selected_language_name}: ", sentiment)

                    # Save results
                    save_button = st.button("Save Results")
                    if save_button:
                        save_as_csv({"summary": [summary], "classification": [classification], "sentiment": [sentiment]})
                        save_as_pdf({"summary": summary, "classification": classification, "sentiment": sentiment})

    elif input_type == "URL":
        url_input = st.text_input("🔗 Enter a URL")

        if st.button("✨ Summarize and Analyze URL"):
            if url_input:
                with st.spinner("Processing..."):
                    # Scrape and summarize
                    text = scrape_website(url_input)
                    summary = text_summary(text)
                    metadata = extract_metadata(url_input)

                    # Display results
                    st.write("📄 URL Summary: ", summary)
                    st.write("📝 URL Metadata: ", metadata)

    elif input_type == "File":
        file_input = st.file_uploader("📂 Upload a file", type=["txt", "pdf", "docx"])

        if file_input is not None:
            # Handle text extraction from file
            text = extract_text_from_file(file_input)
            summary = text_summary(text)
            st.write("📄 File Summary: ", summary)

# Run the app
if __name__ == "__main__":
    main()
