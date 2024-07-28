from flask import Flask, request, render_template
from txtai.pipeline import Summary
from googletrans import Translator
import pytesseract
from PIL import Image
import requests
from bs4 import BeautifulSoup
import pdfplumber
from docx import Document
import os

# Initialize the Flask app
app = Flask(__name__)

# Initialize the summarizer and translator
summary = Summary()
translator = Translator()

# Languages dictionary
languages = {
    "Afrikaans": "af", "Albanian": "sq", "Amharic": "am", "Arabic": "ar", "Armenian": "hy", 
    "Azerbaijani": "az", "Basque": "eu", "Belarusian": "be", "Bengali": "bn", "Bosnian": "bs", 
    "Bulgarian": "bg", "Catalan": "ca", "Chinese (Simplified)": "zh-CN", "Chinese (Traditional)": "zh-TW", 
    "Croatian": "hr", "Czech": "cs", "Danish": "da", "Dutch": "nl", "English": "en", "Esperanto": "eo", 
    "Estonian": "et", "Finnish": "fi", "French": "fr", "Galician": "gl", "Georgian": "ka", 
    "German": "de", "Greek": "el", "Gujarati": "gu", "Haitian Creole": "ht", "Hausa": "ha", 
    "Hebrew": "he", "Hindi": "hi", "Hungarian": "hu", "Icelandic": "is", "Igbo": "ig", 
    "Indonesian": "id", "Irish": "ga", "Italian": "it", "Japanese": "ja", "Javanese": "jv", 
    "Kannada": "kn", "Kazakh": "kk", "Khmer": "km", "Kinyarwanda": "rw", "Korean": "ko", 
    "Kurdish": "ku", "Kyrgyz": "ky", "Lao": "lo", "Latvian": "lv", "Lithuanian": "lt", 
    "Luxembourgish": "lb", "Macedonian": "mk", "Malagasy": "mg", "Malay": "ms", "Malayalam": "ml", 
    "Maltese": "mt", "Maori": "mi", "Marathi": "mr", "Mongolian": "mn", "Nepali": "ne", 
    "Norwegian": "no", "Pashto": "ps", "Persian": "fa", "Polish": "pl", "Portuguese": "pt", 
    "Punjabi": "pa", "Romanian": "ro", "Russian": "ru", "Samoan": "sm", "Scots Gaelic": "gd", 
    "Serbian": "sr", "Sesotho": "st", "Shona": "sn", "Sindhi": "sd", "Sinhala": "si", 
    "Slovak": "sk", "Slovenian": "sl", "Somali": "so", "Spanish": "es", "Sundanese": "su", 
    "Swahili": "sw", "Swedish": "sv", "Tagalog": "tl", "Tajik": "tg", "Tamil": "ta", 
    "Tatar": "tt", "Telugu": "te", "Thai": "th", "Turkish": "tr", "Ukrainian": "uk", 
    "Urdu": "ur", "Uzbek": "uz", "Vietnamese": "vi", "Welsh": "cy", "Xhosa": "xh", 
    "Yoruba": "yo", "Zulu": "zu"
}

# Function to extract text from URL
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    return "\n".join([p.get_text() for p in paragraphs])

# Function to extract text from PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages])

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to extract text from image
def extract_text_from_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)

# Function to save summary to history
def save_summary(summary):
    with open("summary_history.txt", "a", encoding="utf-8") as f:
        f.write(summary + "\n\n")

# Function to load summary history
def load_summary_history():
    if os.path.exists("summary_history.txt"):
        with open("summary_history.txt", "r", encoding="utf-8") as f:
            return f.read().split("\n\n")
    return []

@app.route("/", methods=["GET", "POST"])
def index():
    summary_result = ""
    if request.method == "POST":
        choice = request.form.get("choice")
        language_code = request.form.get("language", "en")

        if choice == "text":
            text = request.form.get("text")
            summary_result = summary(text)
            if language_code and language_code != "en":
                summary_result = translator.translate(summary_result, dest=language_code).text
            save_summary(summary_result)
        elif choice == "url":
            url = request.form.get("url")
            text = extract_text_from_url(url)
            summary_result = summary(text)
            if language_code and language_code != "en":
                summary_result = translator.translate(summary_result, dest=language_code).text
            save_summary(summary_result)
        elif choice == "document":
            file = request.files.get("file")
            if file.filename.endswith(".pdf"):
                text = extract_text_from_pdf(file)
            elif file.filename.endswith(".docx"):
                text = extract_text_from_docx(file)
            else:
                text = file.read().decode("utf-8")
            summary_result = summary(text)
            if language_code and language_code != "en":
                summary_result = translator.translate(summary_result, dest=language_code).text
            save_summary(summary_result)
        elif choice == "image":
            file = request.files.get("image")
            text = extract_text_from_image(file)
            summary_result = summary(text)
            if language_code and language_code != "en":
                summary_result = translator.translate(summary_result, dest=language_code).text
            save_summary(summary_result)
        elif choice == "history":
            summary_result = load_summary_history()

    return render_template("index.html", summary=summary_result, languages=languages, summary_history=load_summary_history())

@app.route("/clear_input", methods=["POST"])
def clear_input():
    return render_template("index.html", summary="", languages=languages, summary_history=load_summary_history())

@app.route("/clear_history", methods=["POST"])
def clear_history():
    if os.path.exists("summary_history.txt"):
        os.remove("summary_history.txt")
    return render_template("index.html", summary="", languages=languages, summary_history=[])

if __name__ == "__main__":
    app.run(debug=True)
