import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crewai import LLM
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import docx
from typing import List
import time
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Utility Functions
def llm_predict_with_backoff(prompt: str, max_retries: int = 5, initial_delay: float = 1.0) -> str:
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            time.sleep(2.0)  # Simulate delay or rate limiting
            response = unified_llm.call(prompt)
            return response
        except Exception as e:
            logging.warning(f"Rate limit error on attempt {attempt + 1}/{max_retries}: {e}")
            time.sleep(delay)
            delay *= 2
    logging.error("Exceeded maximum retry attempts for LLM call")
    raise Exception("Exceeded maximum retry attempts for LLM call")

def extract_text_from_file(file_path: str) -> str:
    """Extract text from various file types."""
    text = ""
    fname = file_path.lower()
    try:
        if fname.endswith('.txt'):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif fname.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif fname.endswith(('.xlsx', '.xls')):
            text = extract_text_from_excel(file_path)
        elif fname.endswith('.csv'):
            text = extract_text_from_csv(file_path)
        elif fname.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            text = extract_text_from_image(file_path)
        elif fname.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        else:
            text = f"Unsupported file type: {file_path}"
    except Exception as e:
        return f"Error extracting text from {file_path}: {e}"
    if not text.strip():
        return f"Warning: No usable text extracted from {file_path}"
    return text

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF, using OCR for image-based pages."""
    text = ""
    with open(file_path, 'rb') as file_content:
        reader = PdfReader(file_content)
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if len(page_text.strip()) > 50:  # Threshold to determine if text-based
                text += f"Page: {page_num}\n{page_text}\n\n"
            else:
                # Use OCR for image-based pages or scanned documents
                images = convert_from_path(file_path, first_page=page_num, last_page=page_num)
                for img in images:
                    ocr_text = pytesseract.image_to_string(img)
                    text += f"Page: {page_num}\n{ocr_text}\n\n"
    return text

def extract_text_from_excel(file_path: str) -> str:
    """Extract text from Excel files, including all sheets."""
    text = ""
    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        text += f"Sheet: {sheet_name}\n{df.to_csv(index=False)}\n\n"
    return text

def extract_text_from_csv(file_path: str) -> str:
    """Extract text from CSV files."""
    df = pd.read_csv(file_path)
    return df.to_csv(index=False)

def extract_text_from_image(file_path: str) -> str:
    """Extract text from images using OCR."""
    image = Image.open(file_path)
    return pytesseract.image_to_string(image)

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from Word documents."""
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# DocumentStore Class
class DocumentStore:
    def __init__(self, use_embeddings: bool = True, chunk_size: int = 500):
        self.use_embeddings = use_embeddings and EMBEDDING_MODEL is not None
        self.chunk_size = chunk_size
        self.docs = []  # List of (short_name, chunk, original_name)
        self.embeddings = []
        self.short_to_original = {}
        self.original_to_short = {}
        self.file_counter = 1

    def _generate_short_name(self, original_name: str) -> str:
        short_name = f"File_{self.file_counter}"
        self.file_counter += 1
        self.short_to_original[short_name] = original_name
        self.original_to_short[original_name] = short_name
        return short_name

    def add_document_text(self, text: str, original_name: str):
        """Add a document to the store by chunking its text."""
        logging.info(f"Adding file to store: {original_name}")
        short_name = self._generate_short_name(original_name)
        chunks = self._chunk_text(text)
        for chunk in chunks:
            self.docs.append((short_name, chunk, original_name))
            if self.use_embeddings:
                emb = EMBEDDING_MODEL.encode(chunk, convert_to_tensor=True)
                self.embeddings.append(emb)
        logging.debug(f"Added document '{original_name}' with {len(chunks)} chunks.")

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end
        return chunks

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[tuple]:
        """Retrieve the most relevant chunks for a query."""
        try:
            from rapidfuzz import fuzz
            best_score = 0
            best_original = None
            for orig in self.original_to_short:
                score = fuzz.partial_ratio(orig.lower(), query.lower())
                if score > best_score:
                    best_score = score
                    best_original = orig
            if best_score > 70:
                short_name = self.original_to_short[best_original]
                return self._retrieve_from_short_name(short_name, query, top_k)
        except ImportError:
            for orig in self.original_to_short:
                if orig.lower() in query.lower():
                    short_name = self.original_to_short[orig]
                    return self._retrieve_from_short_name(short_name, query, top_k)
        if self.use_embeddings and self.embeddings:
            query_emb = EMBEDDING_MODEL.encode(query, convert_to_tensor=True)
            embeddings_tensor = torch.stack(self.embeddings)
            scores = torch.nn.functional.cosine_similarity(query_emb, embeddings_tensor)
            top_results = scores.topk(k=min(top_k, len(scores))).indices.tolist()
            return [self.docs[idx] for idx in top_results]
        # Fallback to keyword matching
        tokens_query = set(query.lower().split())
        def score(item: tuple) -> int:
            return len(tokens_query.intersection(set(item[1].lower().split())))
        ranked = sorted(self.docs, key=score, reverse=True)
        return ranked[:top_k]

    def _retrieve_from_short_name(self, short_name: str, query: str, top_k: int) -> List[tuple]:
        relevant_docs = [(sn, chunk, oname) for (sn, chunk, oname) in self.docs if sn == short_name]
        return relevant_docs[:top_k] if relevant_docs else [("", "", f"No chunks found for file {short_name}")]

    def get_all_files(self) -> List[str]:
        """Return a list of all original file names."""
        return list(set(original_name for _, _, original_name in self.docs))

# RAGChatBot Class
class RAGChatBot:
    def __init__(self, doc_store: DocumentStore):
        self.doc_store = doc_store
        self.conversation_history = ""

    def chat(self, user_query: str, top_k: int = 3) -> str:
        """Handle chat interactions with the user."""
        if "list all the files present in the context" in user_query.lower():
            all_files = self.doc_store.get_all_files()
            response = "The files present in the context are:\n" + "\n".join([f"* {fname}" for fname in all_files]) if all_files else "No files found."
        else:
            relevant_chunks = self.doc_store.retrieve_relevant_chunks(user_query, top_k=top_k)
            context_block = "\n\n".join([f"From {original_name}:\n{chunk}" for _, chunk, original_name in relevant_chunks])
            prompt = (
                "You are a helpful assistant with access to the following document excerpts:\n"
                f"{context_block}\n\n"
                f"User: {user_query}\nAssistant:"
            )
            response = llm_predict_with_backoff(prompt)
        self.conversation_history += f"\nUser: {user_query}\nAssistant: {response}"
        return response

# FastAPI Setup
app = FastAPI()

# Initialize LLM and Embedding Model
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set in environment variables")
unified_llm = LLM(model="gemini/gemini-2.0-flash", api_key=GEMINI_API_KEY, temperature=0.1)
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize DocumentStore and Load Documents
doc_store = DocumentStore(use_embeddings=True)
documents_dir = "documents"
if not os.path.exists(documents_dir):
    os.makedirs(documents_dir)
for filename in os.listdir(documents_dir):
    file_path = os.path.join(documents_dir, filename)
    if os.path.isfile(file_path):
        text = extract_text_from_file(file_path)
        doc_store.add_document_text(text, filename)
        logging.info(f"Loaded document: {filename}")

# Initialize Chatbot
chatbot = RAGChatBot(doc_store)

# API Request Model
class ChatRequest(BaseModel):
    query: str

# Chat Endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = chatbot.chat(request.query)
        return {"response": response}
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=5000)
