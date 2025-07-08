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
import re

app = Flask(__name__)
app.secret_key = "your-secret-key"


#ordner für uploads und vektor
UPLOAD_FOLDER = "uploads"
VECTOR_FOLDER = "tmp/faiss_index"
JSON_FOLDER = "jsons"
os.makedirs(JSON_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("tmp", exist_ok=True)

question_cache = {}

#api konfiguration
config = configparser.ConfigParser()
config.read("config.ini")
API_KEY = config["DEFAULT"]["KEY"]
API_URL = config["DEFAULT"]["ENDPOINT"] + "/chat/completions"
MODEL = "meta-llama-3.1-8b-instruct"

#pdf-text extrahieren
def extract_documents_from_pdf(filepath):
    import fitz  # PyMuPDF

    doc = fitz.open(filepath)
    full_text = ""

    #für die ersten 50 Seiten wird der text als full_text-String gepeichert
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            full_text += text + "\n"

    # Kein OCR mehr! → PDF wird nur analysiert, wenn Text vorhanden ist
    if not full_text.strip():
        print("⚠️ Warnung: Kein Text im PDF gefunden – OCR ist deaktiviert.")

    return full_text

#faiss-Vektorstore erstellen, speichern
def create_and_save_vectorstore(docs, path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(path)

#bestehenden vektorstore laden
def load_vectorstore(path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

#nutzt RAG um zu einer Frage den relevantesten Dokument-Chunk zu finden
def get_context_from_rag(question, vectorstore, k=8):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    return f"Dokumentenauszug:\n{context}" #herausgefundene Kontext wird später an llm geschickt

#frage an das llm mit kontext
def ask_llm(question, context):
    lang = detect(question)
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    if lang == "en":
        system_prompt = (
            "You are a helpful, precise, and polite assistant specialized in analyzing academic and official documents. "
            "You always respond in English. Answer clearly and concisely. "
            "If the user thanks you, respond kindly (e.g., 'You're welcome')."
            "If the user's question asks for multiple items, goals, steps, or points, format your answer as a numbered list with line breaks between points or numbers. "
            "Do not use any formatting like bold, italics, or Markdown. Only provide plain text responses."
        )
    else:
        system_prompt = (
            "Du bist ein hilfsbereiter, präziser und höflicher Assistent für die Analyse akademischer und offizieller Dokumente. "
            "Du antwortest immer auf Deutsch. Antworte klar und kurz. "
            "Wenn sich der Nutzer bedankt, antworte freundlich (z. B. 'Gern geschehen')."
            "Falls die Frage mehrere Punkte, Ziele oder Schritte verlangt, formatiere deine Antwort als nummerierte Liste mit Zeilenumbrüchen zwischen den Punkten oder Nummerierungen. "
            "Bitte verwende keinerlei Formatierungen wie fett, kursiv oder Markdown. Gib deine Antworten nur als reinen Klartext zurück."
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{context}\n\nFrage:\n{question}"}
    ]
    payload = {
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

#flask-Route
@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session: #wenn es noch keinen chat gibt, wird ein leerer Verlauf angelegt
        session["chat_history"] = []

    chat_history = session["chat_history"]
    error = None

    if request.method == "POST": #pdf wird hochgeladen
        pdf_file = request.files.get("pdf")
        question = request.form.get("question")

        if pdf_file and pdf_file.filename.lower().endswith(".pdf"): #geprüft ob es ein pdf ist
            try:
                filename = pdf_file.filename
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                pdf_file.save(filepath)
                session["pdf_filename"] = filename

                #  PDF-Text extrahieren
                full_text = extract_documents_from_pdf(filepath)

                #  In Chunks umwandeln (LangChain Documents)
                splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=20)
                documents = splitter.create_documents([full_text])

                #  Vectorstore erstellen
                session["chat_history"] = []
                if os.path.exists(VECTOR_FOLDER):
                    shutil.rmtree(VECTOR_FOLDER)
                create_and_save_vectorstore(documents, VECTOR_FOLDER)

            except Exception as e:
                error = str(e)

        #wenn pdf, Frage und Vektorstore existieren
        if question and "pdf_filename" in session and os.path.exists(VECTOR_FOLDER):
            try: #vektorstore wird geladen um relevante Inhalte zu finden
                vectorstore = load_vectorstore(VECTOR_FOLDER)
                if question in question_cache: #frage wurde schonmal gestellt
                    answer = question_cache[question]
                else:
                    context = get_context_from_rag(question, vectorstore)
                    answer = ask_llm(question, context)
                    question_cache[question] = answer #antwort der Frage wird gespeichert

                chat_history.append({"question": question, "answer": answer})
                session["chat_history"] = chat_history

            except Exception as e:
                error = str(e)

    return render_template("index.html", chat_history=chat_history, error=error)

#pdf herunterladen
@app.route("/get_pdf")
def get_pdf():
    if "pdf_filename" in session:
        filepath = os.path.join(UPLOAD_FOLDER, session["pdf_filename"])
        if os.path.exists(filepath):
            return send_file(filepath, mimetype="application/pdf")
    return "Keine PDF hochgeladen.", 404

#Löscht Chatverlauf und gespeicherte Daten
@app.route("/reset", methods=["GET"])
def reset():
    session.clear()
    if os.path.exists(VECTOR_FOLDER):
        shutil.rmtree(VECTOR_FOLDER)
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    return redirect("/")

#Löscht Chatverlauf
@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session["chat_history"] = []
    return redirect("/")

#Chatverlauf als Textdatei herunterladen
@app.route("/download_chat", methods=["POST"])
def download_chat():
    history = session.get("chat_history", [])
    output = io.StringIO()
    for entry in history:
        output.write("Du: " + entry["question"] + "\n")
        output.write("Assistent: " + entry["answer"] + "\n\n")
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), as_attachment=True, download_name="chatverlauf.txt", mimetype="text/plain")

#JSON herunterladen
@app.route("/download_key_values", methods=["POST"])
def download_key_values():
    #prüfen, ob json schon einmal geladen worden ist
    json_filename = f"{os.path.splitext(session['pdf_filename'])[0]}.json"
    json_path = os.path.join(JSON_FOLDER, json_filename)

    # Wenn es schon existiert, direkt zurückgeben
    if os.path.exists(json_path):
        return send_file(json_path, as_attachment=True, download_name="key_values.json", mimetype="application/json")
    #prüfen, ob pdf hochgeladen wurde
    if "pdf_filename" not in session or not os.path.exists(VECTOR_FOLDER):
        return "Bitte lade zuerst eine PDF hoch!", 400
    #Schlüsselbegriffe, die herausgefunden werden sollen
    key_list = [
        "name", "CO2", "NOX", "Number_of_Electric_Vehicles", "Impact",
        "Risks", "Opportunities", "Strategy", "Actions", "Adopted_policies", "Targets"
    ]
    vectorstore = load_vectorstore(VECTOR_FOLDER)
    context = ""
    for doc in vectorstore.similarity_search("summary", k=10):
        context += doc.page_content + "\n\n"
    prompt = f"""
You are an AI assistant specialized in analyzing sustainability reports.

Respond with exactly one valid JSON object matching the following keys. 
Do not add any explanations, introductory text, or markdown code blocks like ```json. 
Do not include comments inside the JSON. Only provide valid JSON.
{key_list}
**key explanations:**
- "CO2": How much tons of CO2 did the company produce or reduce?
- "NOX": How much NOX emissions were reported or discussed?
- "Number_of_Electric_Vehicles": How many electric vehicles does the company operate?
- "Impact": What positive or negative environmental or social impacts are described?
- "Risks": What risks are mentioned related to sustainability or climate?
- "Opportunities": What opportunities does the company identify regarding sustainability or climate action?
- "Strategy": What overall strategy does the company present for sustainability?
- "Actions": What concrete actions or steps has the company taken or planned?
- "Adopted_policies": Which policies, standards, or certifications has the company adopted?
- "Targets": What specific goals or targets has the company set (e.g., emission reductions, deadlines)?

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
        text = data["choices"][0]["message"]["content"].strip() #generierten Text holen
        try:
            json_start = text.find('{') #findet den json teil (evtl. fügt llm noch text dazu)
            json_end = text.rfind('}') + 1
            json_str = text[json_start:json_end]
            key_values = json.loads(json_str)
            key_values["name"] = session["pdf_filename"] #json name direkt auf pdf namen setzen
        except Exception:
            key_values = {"error": "Fehler beim Parsen", "response": text}
    else:
        key_values = {"error": "Fehler vom LLM", "status_code": response.status_code, "response": response.text}
    json_data = json.dumps(key_values, indent=2)
    #speichert json
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_data)

    return send_file(io.BytesIO(json_data.encode()), as_attachment=True, download_name="key_values.json", mimetype="application/json")

if __name__ == "__main__":
    app.run(debug=True)
