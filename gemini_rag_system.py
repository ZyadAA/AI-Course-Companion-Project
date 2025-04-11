import requests
from pypdf import PdfReader
import os
import re
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
from typing import List, Dict
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)
# Enable CORS only for the specific HTML frontend
#CORS(app, resources={r"/api/*": {"origins": "*"}})


# Download PDF from a URL and save to local path
def download_pdf(url: str, save_path: str):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# Extract text from PDF
def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Split text into chunks
def split_text(text: str) -> List[str]:
    return [i for i in re.split('\n\n', text) if i.strip()]

# Set and validate the API key for Gemini API
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("Gemini API Key not provided or incorrect. Please provide a valid GEMINI_API_KEY.")
try:
    genai.configure(api_key=gemini_api_key)
    print("API configured successfully with the provided key.")
except Exception as e:
    print("Failed to configure API:", str(e))

# Custom embedding function using Gemini API
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model, content=input, task_type="retrieval_document", title=title)["embedding"]

# Create a Chroma DB and add multiple documents with metadata
def create_chroma_db(documents: List[Dict], path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())

    for doc in documents:
        for i, chunk in enumerate(doc['chunks']):
            metadata = {'source': doc['title'], 'chunk_id': i}
            db.add(documents=[chunk], ids=[f"{doc['title']}-{i}"], metadatas=[metadata])
    return db

# Load a Chroma collection
def load_chroma_collection(path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    return chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())

# Retrieve the most relevant passages based on query
# def get_relevant_passage(query: str, db, n_results: int = 1):
#     results = db.query(query_texts=[query], n_results=n_results)
#     print(results)
#     passages = [(doc[0], meta[0]['source']) for doc, meta in zip(results['documents'], results['metadatas'])]
#     return passages

# Make a prompt for generating an answer
def make_rag_prompt(query: str, relevant_passage: str, source: str):
    escaped_passage = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = f"""You are a helpful bot that uses the reference below to answer questions.
Reference Source: {source}
QUESTION: '{query}'
PASSAGE: '{escaped_passage}'
ANSWER:
"""
    return prompt


# Retrieve the most relevant passages based on the query
def get_relevant_passage(query: str, db, n_results: int):
    results = db.query(query_texts=[query], n_results=n_results)
    return [doc[0] for doc in results['documents']]

# Construct a prompt for the generation model based on the query and retrieved data
def make_rag_prompt(query: str, relevant_passage: str):
    escaped_passage = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below.
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and
strike a friendly and conversational tone.
QUESTION: '{query}'
PASSAGE: '{escaped_passage}'

ANSWER:
"""
    return prompt

# Generate an answer using Gemini Pro API
def generate_answer(prompt: str):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    #initialize chat session
    chat_session = model.start_chat(history=[])

    #get response in chat session
    result = chat_session.send_message(prompt)
    print(chat_session.history)
    print(chat_session.last)
    #result = model.generate_content(prompt)
    #print(prompt)
    return result.text

# Process a user query and generate an answer
def process_query_and_generate_answer(query, db_path, db_name):
    if query == "":
        print("No query provided.")
        return

    db = load_chroma_collection(db_path, db_name)
    relevant_passages = get_relevant_passage(query, db, n_results=1)

    if not relevant_passages:
        print("No relevant information found for the given query.")
        return

    relevant_text= relevant_passages[0]
    final_prompt = make_rag_prompt(query, relevant_text)
    answer = generate_answer(final_prompt)
    print("Generated Answer:", answer)
    return answer


# Create a directory for the Chroma DB if it doesn't exist
db_folder = "chroma_db"
if not os.path.exists(db_folder):
    os.makedirs(db_folder)

# Specify the database name and path
db_name = "ragexperiment"
db_path = os.path.join(os.getcwd(), db_folder)


@app.route('/', methods = ['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/apiwait', methods=['POST'])
def process_question():
    data = request.get_json()
    print(data)
    query = data['query_question']
    print(query)
    # Load multiple PDFs into the Chroma DB
    # pdf_urls = [
    #     "https://services.google.com/fh/files/misc/ai_adoption_framework_whitepaper.pdf",
    #     # Add more URLs if needed
    # ]

    pdf_names = [
        "COP4710-FA24-Topsakal-v240820.pdf",
        "COT4400-FA24-Topsakal-v240825.pdf",
        "L0.COT4400-Intro-Syllabus.pdf",
        "L1.Algorithms-ProofReview.pdf",
        "L3.LoopInvarient.pdf",
        "L4.BigOh.pdf",
        "L5.BigOh-Omega-Theta.pdf",
        "L6.BigOhProperties.pdf",
        "L7.BigOhExamples.pdf"
    ]


    documents = []
    
    # Start the interactive query process
    gemini_response = process_query_and_generate_answer(query, db_path, db_name)
    return jsonify({"gemini_response": gemini_response})


if __name__ == '__main__':
    app.run(debug=True, port = 8080)
    #port = os.environ.get('PORT')
    app.run(host='0.0.0.0')