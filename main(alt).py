from flask import Flask, request, render_template, jsonify
import os
import json
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

API_KEY = 'dein_api_key'
ENDPOINT = 'https://chat-ai.academiccloud.de/api/generate'

stored_text = ''  # wird global gespeichert


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_pdf():
    global stored_text
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'Keine Datei ausgewählt.'})
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # PDF-Text simuliert (ersetze das mit echtem PDF-Parser)
    stored_text = f"Inhalt von Datei {filename}"

    return render_template('index.html', message='PDF erfolgreich hochgeladen.')


@app.route('/chat', methods=['POST'])
def chat():
    global stored_text
    if not stored_text:
        return jsonify({'answer': 'Bitte zuerst ein PDF hochladen.'})

    question = request.form['question']
    payload = {
        "input": f"{stored_text}\n\nFrage: {question}",
        "key": API_KEY
    }
    try:
        response = requests.post(ENDPOINT, json=payload)
        data = response.json()
        answer = data.get("output", "Keine Antwort erhalten.")

        # Antwort in JSON-Datei speichern
        os.makedirs("jsons", exist_ok=True)
        with open("jsons/output.json", "w", encoding="utf-8") as f:
            json.dump({"question": question, "answer": answer}, f, indent=4, ensure_ascii=False)

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Fehler: {str(e)}"})


if __name__ == '__main__':
    app.run(debug=True)
