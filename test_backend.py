import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import logging
import time
import os
import shutil
import threading
import pandas as pd
import pytesseract
from PIL import Image
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
from docx import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

###############################################################################
# Local LLM Setup
###############################################################################

def initialize_local_llm(model_name="naver-clova-ix/donut-base-finetuned-docvqa", temperature=0.1):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    class LocalLLM:
        def __init__(self, model, tokenizer, temperature):
            self.model = model
            self.tokenizer = tokenizer
            self.temperature = temperature
            self.device = device

        def generate_response(self, prompt, max_length=200):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=min(len(prompt)+max_length, 2048),
                    temperature=self.temperature,
                    top_p=0.95,
                    top_k=50,
                    do_sample=True
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

    return LocalLLM(model, tokenizer, temperature)

unified_llm = initialize_local_llm()

###############################################################################
# Utility Functions
###############################################################################

def llm_predict(prompt: str, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = unified_llm.generate_response(prompt, max_length=256)
            return response
        except Exception as e:
            logging.error(f"LLM error: {str(e)[:100]} (Attempt {attempt+1}/3)")
            time.sleep(2)
    return "Error: Maximum retries exceeded"

def extract_text_from_file(file_path):
    fname = file_path.lower()
    try:
        if fname.endswith('.pdf'):
            try:
                text = extract_text(file_path)
            except:
                pages = convert_from_path(file_path, 500)
                text = "\n".join([pytesseract.image_to_string(page) for page in pages])
            return text.strip() or "No text extracted"
        
        elif fname.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
            return df.to_csv(index=False, encoding='utf-8')
        
        elif fname.endswith('.docx'):
            doc = Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])
        
        elif fname.endswith('.csv'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif fname.endswith(('.png', '.jpg', '.jpeg')):
            return pytesseract.image_to_string(Image.open(file_path))
        
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        return f"Error extracting text: {str(e)[:100]}"

###############################################################################
# Document Store
###############################################################################

class DocumentStore:
    def __init__(self, chunk_size=500):
        self.chunk_size = chunk_size
        self.documents = []
        self.file_mapping = {}
        self.file_counter = 1

    def add_document(self, file_path, text):
        short_name = f"File_{self.file_counter}"
        self.file_mapping[short_name] = file_path
        self.file_counter += 1
        
        chunks = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        for chunk in chunks:
            self.documents.append((short_name, chunk, os.path.basename(file_path)))

    def get_relevant_chunks(self, query, top_k=5):
        query_terms = set(query.lower().split())
        scores = []
        for doc in self.documents:
            chunk_terms = set(doc[1].lower().split())
            score = len(query_terms.intersection(chunk_terms))
            scores.append((score, doc))
        scores.sort(reverse=True)
        return [d[1] for d in scores[:top_k]]

###############################################################################
# Main Application Class
###############################################################################

class FileBoxApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Advanced FileBox RAG System")
        self.geometry("1200x800")
        
        self.doc_store = DocumentStore()
        self.chat_history = []
        
        # Create UI components
        self.create_menu()
        self.create_widgets()

    def create_menu(self):
        menu_bar = tk.Menu(self)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Upload Files", command=self.upload_files)
        menu_bar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menu_bar)

    def create_widgets(self):
        # Left Frame for File List
        left_frame = ttk.Frame(self, padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.file_listbox = tk.Listbox(left_frame, width=40)
        self.file_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
        
        # Right Frame for Chat Interface
        right_frame = ttk.Frame(self, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.chat_display = ScrolledText(right_frame, state=tk.DISABLED, height=20)
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=5)
        
        input_frame = ttk.Frame(right_frame)
        input_frame.pack(fill=tk.X)
        
        self.input_entry = ttk.Entry(input_frame, width=50)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_entry.bind("<Return>", self.send_message)
        
        send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        send_button.pack(side=tk.RIGHT)

    def upload_files(self):
        files = filedialog.askopenfilenames()
        for file_path in files:
            filename = os.path.basename(file_path)
            self.file_listbox.insert(tk.END, filename)
            
            text_content = extract_text_from_file(file_path)
            self.doc_store.add_document(file_path, text_content)
            
            logging.info(f"Loaded: {filename} ({len(text_content.split())} tokens)")

    def send_message(self, event=None):
        user_input = self.input_entry.get()
        if user_input.strip():
            self.input_entry.delete(0, tk.END)
            self.update_chat_display(f"User: {user_input}\n")
            
            relevant_chunks = self.doc_store.get_relevant_chunks(user_input, top_k=3)
            context = "\n\n".join([f"[{doc[2]}] {doc[1][:100]}" for doc in relevant_chunks])
            
            prompt = f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer:"
            response = llm_predict(prompt)
            
            self.update_chat_display(f"AI: {response}\n\n")
            self.chat_history.append((user_input, response))

    def update_chat_display(self, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, message)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

if __name__ == "__main__":
    # Install missing packages
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError:
        print("Installing required packages...")
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                              "transformers", "torch", "pdfminer.six",
                              "pytesseract", "python-docx", "openpyxl",
                              "pillow", "tk"])
        print("Restart the application after installation")
        sys.exit()

    # Initialize Tesseract OCR
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update path if needed

    app = FileBoxApp()
    app.mainloop()