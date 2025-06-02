from flask import Flask, request, render_template
import fitz  # PyMuPDF
import configparser
import requests

app = Flask(_name_)

# 📦 Konfiguration aus Datei lesen
config = configparser.ConfigParser()
config.read('config.ini')

API_KEY = config["DEFAULT"]["KEY"]
API_URL = config["DEFAULT"]["ENDPOINT"] + "/chat/completions"
MODEL = "meta-llama-3.1-8b-instruct"


# 🧠 PDF verarbeiten
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# 🤖 LLM fragen
def query_llm(prompt, text):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Du bist ein Assistent für Nachhaltigkeitsberichte."},
            {"role": "user", "content": f"{prompt}\n\nKontext:\n{text}"}
        ],
        "temperature": 0.7
    }

    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Fehler beim LLM: {response.status_code}\n{response.text}"

# 🌐 Flask-Route
@app.route("/", methods=["GET", "POST"])
def index():
    answer = None

    if request.method == "POST":
        pdf_file = request.files["pdf"]
        question = request.form["question"]

        if pdf_file and question:
            text = extract_text_from_pdf(pdf_file)
            answer = query_llm(question, text[:4000])  # nur die ersten 4000 Zeichen

    return render_template("index.html", answer=answer)

if _name_ == "_main_":
    app.run(debug=True)