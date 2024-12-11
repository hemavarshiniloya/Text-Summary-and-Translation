
import streamlit as st
from txtai.pipeline import Summary
from deep_translator import GoogleTranslator
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import xml.etree.ElementTree as ET
import re

# List of languages with their ISO 639-1 codes
languages = {
    "English": "en",
    "Afrikaans": "af",
    "Albanian": "sq",
    "Amharic": "am",
    "Arabic": "ar",
    "Armenian": "hy",
    "Azerbaijani": "az",
    "Basque": "eu",
    "Belarusian": "be",
    "Bengali": "bn",
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Chinese (Simplified)": "zh",
    "Chinese (Traditional)": "zh-TW",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "Esperanto": "eo",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "Galician": "gl",
    "Georgian": "ka",
    "German": "de",
    "Greek": "el",
    "Gujarati": "gu",
    "Haitian Creole": "ht",
    "Hausa": "ha",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Igbo": "ig",
    "Indonesian": "id",
    "Irish": "ga",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jv",
    "Kannada": "kn",
    "Kazakh": "kk",
    "Khmer": "km",
    "Kinyarwanda": "rw",
    "Korean": "ko",
    "Kurdish": "ku",
    "Kyrgyz": "ky",
    "Lao": "lo",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Luxembourgish": "lb",
    "Macedonian": "mk",
    "Malagasy": "mg",
    "Malay": "ms",
    "Malayalam": "ml",
    "Maltese": "mt",
    "Maori": "mi",
    "Marathi": "mr",
    "Mongolian": "mn",
    "Nepali": "ne",
    "Norwegian": "no",
    "Pashto": "ps",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Punjabi": "pa",
    "Romanian": "ro",
    "Russian": "ru",
    "Samoan": "sm",
    "Scots Gaelic": "gd",
    "Serbian": "sr",
    "Sesotho": "st",
    "Shona": "sn",
    "Sindhi": "sd",
    "Sinhala": "si",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Somali": "so",
    "Spanish": "es",
    "Sundanese": "su",
    "Swahili": "sw",
    "Swedish": "sv",
    "Tagalog": "tl",
    "Tajik": "tg",
    "Tamil": "ta",
    "Tatar": "tt",
    "Telugu": "te",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Welsh": "cy",
    "Xhosa": "xh",
    "Yoruba": "yo",
    "Zulu": "zu"
}

# Set page configuration
st.set_page_config(layout="wide")

# Initialize text summarizer
def text_summary(text):
    summary = Summary()
    result = summary(text)
    return result

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[^A-Za-z0-9\s\.]+', '', text)  # Remove unwanted characters
    return text

# Function to translate text
def translate_text(text, target_language):
    try:
        translated = GoogleTranslator(source='auto', target=target_language).translate(text)
        return translated
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text  # Return the original text in case of an error

# Function to read PDF files
def read_pdf(file_path):
    pdf = PdfReader(file_path)
    text = ''
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to read Word documents
def read_word(file_path):
    document = Document(file_path)
    text = ''
    for para in document.paragraphs:
        text += para.text
    return text

# Function to read XML files
def read_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    text = ''
    for elem in root:
        text += elem.text
    return text

# Function to read CSV files
def read_csv(file_path):
    df = pd.read_csv(file_path)
    text = ''
    for col in df.columns:
        text += df[col].to_string()
    return text

# Function to scrape website content
def scrape_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ''
        for para in soup.find_all('p'):
            text += para.get_text() + ' '
        return text.strip()
    except Exception as e:
        st.error(f"Error fetching URL: {str(e)}")
        return ""

# Function to store summary and translation history
def store_summary_and_translation(original, summary, translated):
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "Original Text": original,
        "Summary": summary,
        "Translated Summary": translated
    })

# Main function
def main():
    st.title("📝 Text Summarization and Translation App")
    st.write("This app can summarize text and translate it to various languages.")

    # Language selection in sidebar
    selected_language = st.sidebar.selectbox("🌐 Select a language to translate to", list(languages.keys()), index=0)

    # Input selection
    input_type = st.selectbox("📂 Select input type", ["Text", "File", "URL"], index=0)

    if input_type == "Text":
        # Text input
        text_input = st.text_area("✏️ Enter text to summarize and translate", height=200)

        # Summarize and translate button
        if st.button("✨ Summarize and Translate"):
            if text_input:
                with st.spinner("Processing..."):
                    # Preprocess text
                    text = preprocess_text(text_input)

                    # Summarize text
                    summary = text_summary(text)

                    # Translate summary
                    translated_summary = translate_text(summary, languages[selected_language])

                    # Store history
                    store_summary_and_translation(text, summary, translated_summary)

                    # Display results
                    st.write("📝 Original Text:")
                    st.write(text_input)
                    st.write("📄 Summary:")
                    st.write(summary)
                    st.write("🌍 Translated Summary:")
                    st.write(translated_summary)

                    # Save results
                    st.write("💾 Save Results:")
                    save_button = st.button("Save as Text File")
                    if save_button:
                        with open("results.txt", "w") as f:
                            f.write("Original Text:\n" + text_input + "\n\nSummary:\n" + summary + "\n\nTranslated Summary:\n" + translated_summary)
                        st.write("Results saved to results.txt")

                    # Clear input
                    clear_button = st.button("🧹 Clear Input")
                    if clear_button:
                        st.session_state.clear()

    elif input_type == "File":
        # File uploader
        file_uploaded = st.file_uploader("📥 Upload a file (PDF, Word, XML, CSV)", type=["pdf", "docx", "xml", "csv"], accept_multiple_files=False)

        # Summarize and translate button
        if st.button("✨ Summarize and Translate"):
            if file_uploaded:
                with st.spinner("Processing..."):
                    if file_uploaded.name.endswith('.pdf'):
                        text = read_pdf(file_uploaded)
                    elif file_uploaded.name.endswith('.docx'):
                        text = read_word(file_uploaded)
                    elif file_uploaded.name.endswith('.xml'):
                        text = read_xml(file_uploaded)
                    elif file_uploaded.name.endswith('.csv'):
                        text = read_csv(file_uploaded)

                    # Preprocess text
                    text = preprocess_text(text)

                    # Summarize text
                    summary = text_summary(text)

                    # Translate summary
                    translated_summary = translate_text(summary, languages[selected_language])

                    # Store history
                    store_summary_and_translation(text, summary, translated_summary)

                    # Display results
                    st.write("📝 Original Text:")
                    st.write(text)
                    st.write("📄 Summary:")
                    st.write(summary)
                    st.write("🌍 Translated Summary:")
                    st.write(translated_summary)

                    # Clear input
                    clear_button = st.button("🧹 Clear Input")
                    if clear_button:
                        st.session_state.clear()

    elif input_type == "URL":
        # URL input
        url_input = st.text_input("🔗 Enter a URL")

        # Summarize and translate button
        if st.button("✨ Summarize and Translate"):
            if url_input:
                with st.spinner("Processing..."):
                    # Scrape website content
                    text = scrape_website(url_input)

                    # Preprocess text
                    text = preprocess_text(text)

                    # Summarize text
                    summary = text_summary(text)

                    # Translate summary
                    translated_summary = translate_text(summary, languages[selected_language])

                    # Store history
                    store_summary_and_translation(text, summary, translated_summary)

                    # Display results
                    st.write("📝 Original Text:")
                    st.write(text)
                    st.write("📄 Summary:")
                    st.write(summary)
                    st.write("🌍 Translated Summary:")
                    st.write(translated_summary)

                    # Clear input
                    clear_button = st.button("🧹 Clear Input")
                    if clear_button:
                        st.session_state.clear()

    # Show summary and translation history
    if "history" in st.session_state:
        st.write("📜 Previous Summaries and Translations:")
        for entry in st.session_state.history:
            st.write(f"**Original:** {entry['Original Text']}")
            st.write(f"**Summary:** {entry['Summary']}")
            st.write(f"**Translated Summary:** {entry['Translated Summary']}")
            st.write("---")

        # Clear history button
        clear_history_button = st.button("🗑️ Clear History")
        if clear_history_button:
            st.session_state.history.clear()  # Clear the history
            st.success("History cleared successfully.")

if __name__ == "__main__":
    main()
