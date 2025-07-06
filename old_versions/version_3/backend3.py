from flask import Flask, request, render_template, session, send_file, redirect, url_for
import configparser
import requests
import fitz  # PyMuPDF
import os
import io
import json
import shutil
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from datetime import datetime
from langdetect import detect
from pdf2image import convert_from_path
import pytesseract

#wieso zweimal key?
app = Flask(__name__)#erstellt die flask-app
app.secret_key = "your-secret-key"#??

#speicherordner für hochgeladene pdfs
UPLOAD_FOLDER = "uploads"
VECTOR_FOLDER = "tmp/faiss_index" #was ist faiss-vektor-index?
os.makedirs(UPLOAD_FOLDER, exist_ok=True)#ordner erstellen falls noch nicht existiert
os.makedirs("../tmp", exist_ok=True)#wofür ist tmp gut?

#infos für den apikey
config = configparser.ConfigParser()
config.read("config.ini")
API_KEY = config["DEFAULT"]["KEY"]
API_URL = config["DEFAULT"]["ENDPOINT"] + "/chat/completions"
MODEL = "meta-llama-3.1-8b-instruct"

#installationspfad???
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

#ab hier teil 2: pdf-verarbeitung und textextraktion
#funktion liest text aus dem pdf
def extract_documents_from_pdf(filepath):
    doc = fitz.open(filepath) #doc ist das hochgeladene pdf
    full_text = "" # platzhalter für text im dokument

    for page in doc: #für jede seite im dokument?
        text = page.get_text() #text ist gleich der text auf jeder seite? woher kommt definition gettext?
        if text.strip(): #falls text gefunden wird
            full_text += text + "\n" #text wird ergänzt durch text auf dieser seite

    if not full_text.strip(): #falls kein text gefunden wird
        print("⚠️ Kein Text gefunden – OCR wird aktiviert") #ocr ist texterkennung auf bildern
        try:
            images = convert_from_path(filepath, dpi=100, first_page=1, last_page=20) #seiten werden in bilder verwandelt, aber wieso?? und was sollen die zahlen?
            for image in images: #für jedes bild in den konvertierten bildern
                ocr_text = pytesseract.image_to_string(image, lang='deu') #liest text aus bildern. sprache deutsch??
                full_text += ocr_text + "\n" #text wird ergänzt durch text aus bildern
        except Exception as e: #falls es nicht funktioniert??
            print("❌ OCR-Fehler:", e)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=80) #splittet in chunks a 1000 zeichen. (Klasse, noch leer)
    return [Document(page_content=chunk) for chunk in splitter.split_text(full_text)]#zurückgegeben wird eine liste von textstücken (es wurde gesplittet)
    #jeder abschnitt wird in Document-object verwandelt (weil faiss und langchain mit Document-objekten arbeiten

#teil 3 - langchain vektorstore. was ist faiss?

#verwandelt textchunks in vektoren mithilfe eines embeddings-modells und speichert sie lokal
def create_and_save_vectorstore(docs, path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")#?
    vectorstore = FAISS.from_documents(docs, embeddings)#?
    vectorstore.save_local(path)#?

#lädt den gespeicherten faiss-vektorstore
def load_vectorstore(path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

#findet 8 relevantesten Textstellen zum prompt??
def get_context_from_rag(question, vectorstore, k=8):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    return f"Dokumentenauszug:\n{context}"

#teil 4 - LLM-Anfrage mit Kontext
#
def ask_llm(question, context):
    lang = detect(question) # erkennt sprache
    headers = { # ??
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    if lang == "en": #wenn frage auf englisch
        system_prompt = (
            "You are a helpful, precise, and polite assistant specialized in analyzing academic and official documents. "
            "You always respond in English. Answer clearly and concisely. "
            "If the user thanks you, respond kindly (e.g., 'You're welcome')."
        )
    else: #wenn frage auf deutsch, oder auch andere sprachen
        system_prompt = (
            "Du bist ein hilfsbereiter, präziser und höflicher Assistent für die Analyse akademischer und offizieller Dokumente. "
            "Du antwortest immer auf Deutsch. Antworte klar und kurz. "
            "Wenn sich der Nutzer bedankt, antworte freundlich (z. B. 'Gern geschehen')."
        )
    messages = [#??
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{context}\n\nFrage:\n{question}"}
    ]
    payload = {#??
        "model": MODEL,
        "messages": messages,
        "temperature": 0.3
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        return f"Fehler beim LLM: {response.status_code}\n{response.text}"

#flask-route (web-interface)
@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []
    chat_history = session["chat_history"]
    error = None
    if request.method == "POST":
        pdf_file = request.files.get("pdf")
        question = request.form.get("question")
        if pdf_file and pdf_file.filename.lower().endswith(".pdf"):
            try:
                filename = pdf_file.filename
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                pdf_file.save(filepath)
                session["pdf_filename"] = filename
                docs = extract_documents_from_pdf(filepath)
                session["chat_history"] = []
                if os.path.exists(VECTOR_FOLDER):
                    shutil.rmtree(VECTOR_FOLDER)
                create_and_save_vectorstore(docs, VECTOR_FOLDER)
            except Exception as e:
                error = str(e)
        if question and "pdf_filename" in session and os.path.exists(VECTOR_FOLDER):
            try:
                vectorstore = load_vectorstore(VECTOR_FOLDER)
                context = get_context_from_rag(question, vectorstore)
                answer = ask_llm(question, context)
                chat_history.append({"question": question, "answer": answer})
                session["chat_history"] = chat_history
            except Exception as e:
                error = str(e)
    return render_template("index.html", chat_history=chat_history, error=error)

@app.route("/get_pdf")
def get_pdf(): #download der aktuell hochgeladenen PDF
    if "pdf_filename" in session:
        filepath = os.path.join(UPLOAD_FOLDER, session["pdf_filename"])
        if os.path.exists(filepath):
            return send_file(filepath, mimetype="application/pdf")
    return "Keine PDF hochgeladen.", 404

#buttons auf webseite
@app.route("/reset", methods=["GET"])
def reset(): #Löscht alles (PDF, Chat, Vektoren)
    session.clear()
    if os.path.exists(VECTOR_FOLDER):
        shutil.rmtree(VECTOR_FOLDER)
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    return redirect("/")

@app.route("/clear_chat", methods=["POST"])
def clear_chat():#Löscht nur den Chatverlauf
    session["chat_history"] = []
    return redirect("/")

@app.route("/download_chat", methods=["POST"])
def download_chat():#chat downloaden
    history = session.get("chat_history", [])
    output = io.StringIO()
    for entry in history:
        output.write("Du: " + entry["question"] + "\n")
        output.write("Assistent: " + entry["answer"] + "\n\n")
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), as_attachment=True, download_name="chatverlauf.txt", mimetype="text/plain")

@app.route("/download_key_values", methods=["POST"])
def download_key_values():#json runterladen
    if "pdf_filename" not in session or not os.path.exists(VECTOR_FOLDER):
        return "Bitte lade zuerst eine PDF hoch!", 400
    key_list = [
        "name", "CO2", "NOX", "Number_of_Electric_Vehicles", "Impact",
        "Risks", "Opportunities", "Strategy", "Actions", "Adopted_policies", "Targets"
    ]
    vectorstore = load_vectorstore(VECTOR_FOLDER)
    context = ""
    for doc in vectorstore.similarity_search("summary", k=10):
        context += doc.page_content + "\n\n"
    prompt = f"""
Du bist ein KI-Assistent. Extrahiere die folgenden Schlüsselinformationen aus dem bereitgestellten PDF-Kontext zu Nachhaltigkeitsberichten.
Gib die Antwort als gültiges JSON mit folgenden Keys zurück:

{key_list}

Kontext:
{context}

Achte darauf, dass fehlende Informationen als \"Not mentioned\" ausgegeben werden.
"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Extrahiere Key Values als JSON für Nachhaltigkeitsberichte."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        try:
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            json_str = text[json_start:json_end]
            key_values = json.loads(json_str)
        except Exception:
            key_values = {"error": "Fehler beim Parsen", "response": text}
    else:
        key_values = {"error": "Fehler vom LLM", "status_code": response.status_code, "response": response.text}
    json_data = json.dumps(key_values, indent=2)
    return send_file(io.BytesIO(json_data.encode()), as_attachment=True, download_name="key_values.json", mimetype="application/json")

#app starten
if __name__ == "__main__":
    app.run(debug=True, port=5002)
