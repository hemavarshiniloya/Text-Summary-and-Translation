import streamlit as st
from txtai.pipeline import Summary
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import xml.etree.ElementTree as ET
import os
import pytesseract
from googletrans import Translator
import re
import torch
from PIL import Image
from rake_nltk import Rake
import language_tool_python

# List of languages with their ISO 639-1 codes
languages = {
    "English": "en", 
    # (list truncated for brevity)
}

# Initialize tools
summarizer = Summary()
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
rake = Rake()
tool = language_tool_python.LanguageTool('en-US')

# Set page configuration
st.set_page_config(layout="wide")

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Function to compute text similarity
def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

# Function for text classification
def classify_text(text):
    return classifier(text)

# Function for keyword extraction
def extract_keywords(text):
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()

# Function for grammar and spell check
def grammar_and_spell_check(text):
    matches = tool.check(text)
    return [match.message for match in matches]

# Function to extract text from URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
        text = "\n".join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to extract text from TXT
def extract_text_from_txt(file):
    return file.read().decode("utf-8")

# Function to extract text from HTML
def extract_text_from_html(file):
    soup = BeautifulSoup(file.read(), "html.parser")
    paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    text = "\n".join([p.get_text() for p in paragraphs])
    return text

# Function to extract text from CSV
def extract_text_from_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

# Function to extract text from XML
def extract_text_from_xml(file):
    tree = ET.parse(file)
    root = tree.getroot()
    text = " ".join([elem.text for elem in root.iter() if elem.text])
    return text

# Function to extract text from Image
def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

# Function to save summary to history
def save_summary(summary):
    filename = "summary_history.txt"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(summary + "\n\n")

# Function to load summary history
def load_summary_history():
    filename = "summary_history.txt"
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(filename, "r", encoding="latin1") as f:
                return f.read()
    return ""

# Function to clear summary history
def clear_summary_history():
    filename = "summary_history.txt"
    if os.path.exists(filename):
        os.remove(filename)

# Function to clear input fields based on choice
def clear_input(choice):
    if choice == "Summarize Text":
        st.session_state.text = ""
    elif choice == "Summarize URL":
        st.session_state.url = ""
    elif choice == "Summarize Document":
        st.session_state.uploaded_files = []
    elif choice == "Summarize Text from Clipboard":
        st.session_state.clipboard_text = ""

# Function to validate input
def validate_input(text):
    return bool(text and text.strip())

# Function to translate text using Google Translate API
def translate_text(text, target_language):
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    return translated.text

# Function to download file
def download_file(content, filename):
    st.download_button(label="Download Summary", data=content, file_name=filename, mime="text/plain")

# Main function to run the Streamlit app
def main():
    st.title("Text Analysis App")

    # Language selection
    selected_language = st.sidebar.selectbox("Select Language", options=list(languages.keys()), index=0)

    # Handle choice selection
    choice = st.sidebar.radio("Choose an option", [
        "Summarize Text", "Summarize URL", "Summarize Document", 
        "Summarize Text from Clipboard", "Compare Text Similarity",
        "Classify Text", "Extract Keywords", "Check Grammar and Spelling"
    ])

    # Initialize session state attributes if they don't exist
    if 'text' not in st.session_state:
        st.session_state.text = ""
    if 'url' not in st.session_state:
        st.session_state.url = ""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'clipboard_text' not in st.session_state:
        st.session_state.clipboard_text = ""

    # Handle each choice
    if choice == "Summarize Text":
        st.session_state.text = st.text_area("Enter Text", st.session_state.text)
        maxlength = st.slider("Maximum Summary Length", min_value=50, max_value=1000, value=200)

        if st.button("Summarize"):
            if validate_input(st.session_state.text):
                with st.spinner("Processing..."):
                    text = preprocess_text(st.session_state.text)
                    summary = summarizer(text, maxlength)
                    translated_summary = translate_text(summary, languages[selected_language])
                    
                    st.write("### Summary")
                    st.write(translated_summary)

                    save_summary(translated_summary)
                    download_file(translated_summary, "summary.txt")

    elif choice == "Summarize URL":
        st.session_state.url = st.text_input("Enter URL", st.session_state.url)

        if st.button("Summarize URL"):
            if validate_input(st.session_state.url):
                with st.spinner("Processing..."):
                    text = extract_text_from_url(st.session_state.url)
                    summary = summarizer(text)
                    translated_summary = translate_text(summary, languages[selected_language])
                    
                    st.write("### Summary")
                    st.write(translated_summary)

                    save_summary(translated_summary)
                    download_file(translated_summary, "summary.txt")

    elif choice == "Summarize Document":
        uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)
        st.session_state.uploaded_files = uploaded_files

        if st.button("Summarize Document"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    text = ""
                    for uploaded_file in uploaded_files:
                        if uploaded_file.type == "application/pdf":
                            text += extract_text_from_pdf(uploaded_file)
                        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            text += extract_text_from_docx(uploaded_file)
                        elif uploaded_file.type == "text/plain":
                            text += extract_text_from_txt(uploaded_file)
                        elif uploaded_file.type == "text/html":
                            text += extract_text_from_html(uploaded_file)
                        elif uploaded_file.type == "text/csv":
                            text += extract_text_from_csv(uploaded_file)
                        elif uploaded_file.type == "application/xml":
                            text += extract_text_from_xml(uploaded_file)
                        elif uploaded_file.type.startswith("image/"):
                            text += extract_text_from_image(uploaded_file)
                    
                    summary = summarizer(text)
                    translated_summary = translate_text(summary, languages[selected_language])
                    
                    st.write("### Summary")
                    st.write(translated_summary)

                    save_summary(translated_summary)
                    download_file(translated_summary, "summary.txt")

    elif choice == "Summarize Text from Clipboard":
        st.session_state.clipboard_text = st.text_area("Paste Text from Clipboard", st.session_state.clipboard_text)

        if st.button("Summarize Clipboard Text"):
            if validate_input(st.session_state.clipboard_text):
                with st.spinner("Processing..."):
                    text = preprocess_text(st.session_state.clipboard_text)
                    summary = summarizer(text)
                    translated_summary = translate_text(summary, languages[selected_language])
                    
                    st.write("### Summary")
                    st.write(translated_summary)

                    save_summary(translated_summary)
                    download_file(translated_summary, "summary.txt")

    elif choice == "Compare Text Similarity":
        text1 = st.text_area("Enter First Text")
        text2 = st.text_area("Enter Second Text")

        if st.button("Compare Similarity"):
            if validate_input(text1) and validate_input(text2):
                with st.spinner("Processing..."):
                    similarity = compute_similarity(text1, text2)
                    st.write(f"### Similarity Score: {similarity:.4f}")

    elif choice == "Classify Text":
        text = st.text_area("Enter Text for Classification")

        if st.button("Classify Text"):
            if validate_input(text):
                with st.spinner("Processing..."):
                    result = classify_text(text)
                    st.write("### Classification Result")
                    st.write(result)

    elif choice == "Extract Keywords":
        text = st.text_area("Enter Text for Keyword Extraction")

        if st.button("Extract Keywords"):
            if validate_input(text):
                with st.spinner("Processing..."):
                    keywords = extract_keywords(text)
                    st.write("### Keywords")
                    st.write(", ".join(keywords))

    elif choice == "Check Grammar and Spelling":
        text = st.text_area("Enter Text for Grammar and Spell Check")

        if st.button("Check Text"):
            if validate_input(text):
                with st.spinner("Processing..."):
                    issues = grammar_and_spell_check(text)
                    st.write("### Grammar and Spelling Issues")
                    st.write("\n".join(issues))

    if st.sidebar.button("Clear Input"):
        clear_input(choice)

    if st.sidebar.button("Clear Summary History"):
        clear_summary_history()

    if st.sidebar.button("Load Summary History"):
        history = load_summary_history()
        if history:
            st.write("### Summary History")
            st.write(history)
        else:
            st.write("No summary history found.")

if __name__ == "__main__":
    main()
