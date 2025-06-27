from flask import Flask, request, render_template, session, send_file, redirect, url_for
import configparser
import requests
import fitz  # PyMuPDF
import os
import io
import json
import shutil
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

app = Flask(__name__)
app.secret_key = "your-secret-key"  # Setze einen echten Key für Produktion

UPLOAD_FOLDER = "uploads"
VECTOR_FOLDER = "tmp/faiss_index"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("tmp", exist_ok=True)

# 📥 Konfiguration
config = configparser.ConfigParser()
config.read("config.ini")
API_KEY = config["DEFAULT"]["KEY"]
API_URL = config["DEFAULT"]["ENDPOINT"] + "/chat/completions"
MODEL = "meta-llama-3.1-8b-instruct"

# 📚 PDF zu Dokumenten
def extract_documents_from_pdf(filepath):
    doc = fitz.open(filepath)
    full_text = "\n".join(page.get_text() for page in doc)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return [Document(page_content=chunk) for chunk in splitter.split_text(full_text)]

# 🧠 Vektorstore erstellen und speichern
def create_and_save_vectorstore(docs, path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(path)

# Vektorstore laden
def load_vectorstore(path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

# Kontext abrufen
def get_context_from_rag(question, vectorstore, k=3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    results = retriever.invoke(question)
    return "\n\n".join(doc.page_content for doc in results)

# Anfrage an LLM
def ask_llm(question, context):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Du bist ein Assistent für Nachhaltigkeitsberichte."},
            {"role": "user", "content": f"Kontext:\n{context}\n\nFrage:\n{question}"}
        ],
        "temperature": 0.7
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            return "⚠️ Keine Antwort vom Modell erhalten."
    else:
        return f"Fehler beim LLM: {response.status_code}\n{response.text}"

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []
    chat_history = session["chat_history"]
    error = None

    if request.method == "POST":
        pdf_file = request.files.get("pdf")
        question = request.form.get("question")

        # --- Erstes Hochladen einer PDF ---
        if pdf_file and pdf_file.filename.lower().endswith(".pdf"):
            try:
                # PDF speichern
                filename = pdf_file.filename
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                pdf_file.save(filepath)
                session["pdf_filename"] = filename
                # Dokumente aus PDF extrahieren und Vectorstore erstellen
                docs = extract_documents_from_pdf(filepath)
                # Leere Chat-History bei neuem Upload
                session["chat_history"] = []
                # Vectorstore speichern
                if os.path.exists(VECTOR_FOLDER):
                    shutil.rmtree(VECTOR_FOLDER)
                create_and_save_vectorstore(docs, VECTOR_FOLDER)
            except Exception as e:
                error = str(e)

        # --- Frage an vorhandene PDF/Vectorstore stellen ---
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

# --- Route für PDF-Vorschau ---
@app.route("/get_pdf")
def get_pdf():
    if "pdf_filename" in session:
        filepath = os.path.join(UPLOAD_FOLDER, session["pdf_filename"])
        if os.path.exists(filepath):
            return send_file(filepath, mimetype="application/pdf")
    return "Keine PDF hochgeladen.", 404

@app.route("/download_chat", methods=["POST"])
def download_chat():
    history = session.get("chat_history", [])
    output = io.StringIO()
    for entry in history:
        output.write("Du: " + entry["question"] + "\n")
        output.write("Assistent: " + entry["answer"] + "\n\n")
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), as_attachment=True, download_name="chatverlauf.txt", mimetype="text/plain")

@app.route("/download_key_values", methods=["POST"])
def download_key_values():
    # Prüfen, ob PDF und Vektorstore vorhanden sind
    if "pdf_filename" not in session or not os.path.exists("tmp/faiss_index"):
        return "Bitte lade zuerst eine PDF hoch!", 400

    # Typische Key Values
    key_list = [
        "name",
        "CO2",
        "NOX",
        "Number_of_Electric_Vehicles",
        "Impact",
        "Risks",
        "Opportunities",
        "Strategy",
        "Actions",
        "Adopted_policies",
        "Targets"
    ]

    # Hole den Kontext aus dem gesamten PDF (Vektorstore, alles zusammenfassen)
    vectorstore = load_vectorstore("tmp/faiss_index")
    # Optional: Hole besonders relevante Chunks (hier einfach alle nehmen)
    context = ""
    for doc in vectorstore.similarity_search("summary", k=10):
        context += doc.page_content + "\n\n"

    # Prompt für das LLM, damit ein JSON mit allen Key Values extrahiert wird
    prompt = f"""
Du bist ein KI-Assistent. Extrahiere die folgenden Schlüsselinformationen aus dem bereitgestellten PDF-Kontext zu Nachhaltigkeitsberichten.
Gib die Antwort als gültiges JSON mit folgenden Keys zurück:

{key_list}

Kontext:
{context}

Achte darauf, dass fehlende Informationen als "Not mentioned" ausgegeben werden.
"""

    # Anfrage an LLM senden
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
        # Versuche, den JSON-Teil herauszulesen (falls der Bot etwas Text drumherum schreibt)
        try:
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            json_str = text[json_start:json_end]
            key_values = json.loads(json_str)
        except Exception:
            # Notfall: alles als Text
            key_values = {"error": "Fehler beim Parsen", "response": text}
    else:
        key_values = {"error": "Fehler vom LLM", "status_code": response.status_code, "response": response.text}

    json_data = json.dumps(key_values, indent=2)
    return send_file(io.BytesIO(json_data.encode()), as_attachment=True, download_name="key_values.json", mimetype="application/json")


@app.route("/reset", methods=["GET"])
def reset():
    session.clear()
    # Lösche hochgeladenes PDF und Vectorstore
    if os.path.exists(VECTOR_FOLDER):
        shutil.rmtree(VECTOR_FOLDER)
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
