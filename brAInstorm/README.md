# AI-based Applications
Team BrAInstorm
# PDF Question Answering Web App

This application allows users to upload a pdf, analyzes it and answers questions about it.
The answers are based on the use of Retrieval-Augmented Generation and the use of an external Large-Language-Model.
---

##  Installation Guide

###  Requirements

- Python 3.13 installed
- `pip` installed
- We recommend using PyCharm for this project


#### Run the following Command for installing the required packages:
```python
pip install -r requirements.txt
```
---
## Running Guide
#### 1. Insert API-Key
Insert your API-Key and Endpoint in the file config.ini
#### 2. Start the project
After all packages are installed successfully, you can start the project by running this command:
```python
python backend5.py
```
#### 3. Try the application
Now you can upload a PDF-File, and ask questions about it.
You can also download the key values, download or clear the chat or start everything over.
---
## Video of our project
Here you can see a video of how the application is used.
It shows the process of uploading a PDF, asking questions about it and the extra features.

https://youtu.be/ZTpCEU-y3g8

---
## System Architecture

---
## Development Process
In this section, you can read about our stages of developing this application.
Along with this process, which shows our work on the back- and frontend, we also obtained our GitHub Repository and our SCRUM board.
#### 1. Working LLM Connection
First, we started with a minimal prototype:
a simple web interface where users could only upload a PDF and ask a single question.
We connected the backend to an LLM which received the entire PDF text along with the question.
#### 2. RAG Integration
In the second stage we concentrated on implementing RAG into our backend. 
We build a few functions for this, so that only the relevant chunks would be sent to the LLM which enhanced performance and speed significantly.
#### 3. Frontend Improvements
We redesigned the web interface to support the experience of chatting with the PDF, so that users could ask more than one question.
We also added clearer formatting and improved the Layout using HTML, JavaScript and CSS.
#### 4. Extra Features
After that, we added useful features like downloading the chat history, extracting structured information as JSON, and clearing the chat.
### 5. Refining and Optimization
In the last stage, we optimized the backend for speed and robustness, cleaned up the code structure and polished the frontend to deliver a more professional user experience.

---
## Technology 

---
## Problems

---
## Contributions


