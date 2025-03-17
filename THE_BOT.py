import os
import math
import time
import asyncio
import json
import re
import tempfile
import concurrent.futures
import traceback
from pathlib import Path

import chromadb
import ollama
import gradio as gr
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

# ---------------------- Ensure a Running Event Loop ----------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# ---------------------- System Prompt ----------------------
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

Context will be passed as "Context:"
User question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response using clear language, paragraphs, bullet points, and headings where appropriate.

Important: Base your response solely on the provided context without introducing external information.
"""

# ---------------------- Utility Functions ----------------------
def sanitize_id(name: str) -> str:
    return re.sub(r'\W+', '_', name)

# ---------------------- Dynamic Chunking ----------------------
def dynamic_split_documents(doc: Document) -> list:
    text = doc.page_content
    total_len = len(text)
    if total_len <= 1000:
        return [doc]
    num_chunks = math.ceil(total_len / 1000)
    chunk_size = math.ceil(total_len / num_chunks)
    overlap = min(100, chunk_size // 5)
    splits = []
    start = 0
    while start < total_len:
        end = min(start + chunk_size, total_len)
        chunk_text = text[start:end]
        splits.append(Document(page_content=chunk_text, metadata=doc.metadata))
        if end == total_len:
            break
        start = end - overlap
    return splits

# ---------------------- Document Processing ----------------------
def process_file(uploaded_file) -> tuple:
    print(f"Starting processing for file: {uploaded_file}")
    # Check if the uploaded_file has a 'read' method; if not, assume it's a file path.
    if hasattr(uploaded_file, "read"):
        file_obj = uploaded_file
    else:
        file_obj = open(uploaded_file, "rb")
    try:
        file_bytes = file_obj.read()
        print(f"Read {len(file_bytes)} bytes from file.")
    finally:
        if file_obj is not uploaded_file:
            file_obj.close()
    extension = os.path.splitext(uploaded_file.name if hasattr(uploaded_file, "name") else uploaded_file)[1].lower()
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=extension, delete=False)
    try:
        temp_file.write(file_bytes)
        temp_file.close()
        print(f"Temporary file created: {temp_file.name}")

        pipeline_options = PdfPipelineOptions()
        pipeline_options.ocr_options = EasyOcrOptions()
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True

        if extension == ".pdf":
            input_format = InputFormat.PDF
        elif extension == ".docx":
            input_format = InputFormat.DOCX
        elif extension in [".png", ".jpg", ".jpeg"]:
            input_format = InputFormat.IMAGE
        else:
            return [], f"Unsupported file type: {extension}"

        print(f"Using input format: {input_format}")
        converter = DocumentConverter(
            format_options={input_format: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        print("Starting document conversion...")
        start_time = time.time()
        result = converter.convert(temp_file.name)
        parse_time = time.time() - start_time
        print(f"Document conversion completed in {parse_time:.2f} seconds")
        docling_doc = result.document

        doc_dict = docling_doc.dict()
        doc_details = {
            "pages": len(docling_doc.pages.keys()),
            "text_elements": len(doc_dict.get('texts', [])),
            "tables": len(doc_dict.get('tables', [])),
            "pictures": len(doc_dict.get('pictures', []))
        }
        print(f"Document details: {doc_details}")

        if hasattr(docling_doc, "pages") and docling_doc.pages:
            sorted_pages = sorted(docling_doc.pages.items(), key=lambda x: int(x[0]))
            full_text = ""
            page_refs = {}
            for page_num, content in sorted_pages:
                marker = f"--- Page {page_num} ---\n"
                full_text += marker + str(content) + "\n"  # Convert content to string
                page_refs[page_num] = f"[Page {page_num}](#page-{page_num})"
            page_refs_str = json.dumps(page_refs)
        else:
            full_text = docling_doc.export_to_markdown()
            page_refs_str = json.dumps({})

        metadata = {
            "source": uploaded_file.name if hasattr(uploaded_file, "name") else uploaded_file,
            "doc_details": json.dumps(doc_details),
            "page_refs": page_refs_str
        }
        doc = Document(page_content=full_text, metadata=metadata)
        splits = dynamic_split_documents(doc)
        details_str = f"""
**Parsing time:** {parse_time:.2f} seconds
**Doc_name:** {docling_doc.origin.filename}
**Doc_type:** {docling_doc.origin.mimetype}
**Doc_pages_nos.:** {doc_details["pages"]}
**Doc_text_elements:** {doc_details["text_elements"]}
**Doc_tables_nos.:** {doc_details["tables"]}
**Doc_images_nos.:** {doc_details["pictures"]}
"""
        print("File processed successfully.")
        return splits, details_str
    except Exception as e:
        error_details = traceback.format_exc()
        error_msg = f"Failed to process {uploaded_file}: {e}\n{error_details}"
        print(error_msg)
        return [], error_msg
    finally:
        try:
            os.unlink(temp_file.name)
            print(f"Temporary file {temp_file.name} removed.")
        except Exception as cleanup_error:
            print(f"Error removing temporary file {temp_file.name}: {cleanup_error}")

# ---------------------- ChromaDB Setup ----------------------
def get_vector_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )

# ---------------------- Batch Upsert ----------------------
def add_to_vector_collection(all_splits: list, file_name: str):
    collection = get_vector_collection()
    documents = []
    metadatas = []
    ids = []
    sanitized_file_name = sanitize_id(file_name)
    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{sanitized_file_name}_{idx}")
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    print(f"Upserted {len(documents)} chunks for file {file_name}")

# ---------------------- Deletion Functions ----------------------
def delete_file_from_vector_store(file_name: str):
    collection = get_vector_collection()
    collection.delete(where={"source": file_name})
    print(f"Deleted data for file {file_name} from vector store.")

def clear_vector_store():
    collection = get_vector_collection()
    all_data = collection.get()
    if "ids" in all_data and all_data["ids"]:
        collection.delete(ids=all_data["ids"])
        print("Vector store cleared.")
    else:
        print("Vector store already empty.")

# ---------------------- Query & LLM Calls ----------------------
def query_collection(prompt: str, n_results: int = 10, file_filter: list = None):
    collection = get_vector_collection()
    where_clause = {"source": {"$in": file_filter}} if file_filter else None
    print(f"Querying vector store with prompt: {prompt}")
    return collection.query(query_texts=[prompt], n_results=n_results, where=where_clause)

def call_llm(context: str, prompt: str):
    print("Calling LLM with provided context and prompt...")
    response = ollama.chat(
        model="llama3.2-vision:latest",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\nQuestion: {prompt}"},
        ],
        stream=True
    )
    for chunk in response:
        if not chunk.get("done", True):
            yield chunk["message"]["content"]
        else:
            break

def re_rank_cross_encoders(query: str, documents: list, metadatas: list) -> tuple:
    if not documents:
        return "", []
    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("Re-ranking retrieved documents...")
    ranks = encoder_model.rank(query, documents, top_k=3)
    relevant_text = ""
    relevant_indices = []
    for rank in ranks:
        idx = rank["corpus_id"]
        source = metadatas[idx].get("source", "Unknown Source")
        details_str = metadatas[idx].get("doc_details", "No details")
        match = re.search(r"--- Page (\d+) ---", documents[idx])
        ref_link = f"[Page {match.group(1)}](#page-{match.group(1)})" if match else "No page reference"
        relevant_text += f"\n[Source: {source} | Details: {details_str} | Reference: {ref_link}]\n{documents[idx]}\n"
        relevant_indices.append(idx)
    return relevant_text, relevant_indices

def get_all_file_names() -> list:
    collection = get_vector_collection()
    data = collection.get()
    file_names = {meta["source"] for meta in data.get("metadatas", []) if "source" in meta}
    return list(file_names)

# ---------------------- Parallel Processing Function ----------------------
def process_and_upsert_file(uploaded_file) -> tuple:
    name = uploaded_file.name if hasattr(uploaded_file, "name") else uploaded_file
    normalized_name = name.translate(str.maketrans({"-": "_", ".": "_", " ": "_"}))
    stored_files = get_all_file_names()
    if name in stored_files:
        message = f"File {name} is already loaded in the vector store. Skipping upload."
        print(message)
        return normalized_name, False, message
    splits, process_message = process_file(uploaded_file)
    if splits:
        add_to_vector_collection(splits, normalized_name)
        final_message = f"{process_message}\nData from {name} added to vector store."
        return normalized_name, True, final_message
    else:
        return normalized_name, False, process_message

# ---------------------- Gradio Interface Functions ----------------------
def process_files(uploaded_files):
    message = ""
    if not uploaded_files:
        return "No files uploaded.", gr.update(choices=get_all_file_names()), gr.update(choices=get_all_file_names()), get_details()
    for file in uploaded_files:
        normalized_name, success, msg = process_and_upsert_file(file)
        message += f"{msg}\n"
    file_list = get_all_file_names()
    return message, gr.update(choices=file_list), gr.update(choices=file_list), get_details()

def delete_file(file_name):
    if file_name:
        delete_file_from_vector_store(file_name)
        file_list = get_all_file_names()
        return f"Deleted file {file_name}.", gr.update(choices=file_list), gr.update(choices=file_list), get_details()
    else:
        file_list = get_all_file_names()
        return "Please select a file to delete.", gr.update(choices=file_list), gr.update(choices=file_list), get_details()

def clear_vector_store_fn():
    clear_vector_store()
    file_list = get_all_file_names()
    return "Vector store cleared!", gr.update(choices=file_list), gr.update(choices=file_list), get_details()

def get_details():
    file_names = get_all_file_names()
    details = "\n".join(file_names) if file_names else "No files stored."
    return details

def answer_question(prompt, file_filter):
    results = query_collection(prompt, file_filter=file_filter if file_filter else None)
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    if not docs or not metas or len(docs[0]) == 0:
        yield "No relevant documents found. Please try a different query or upload documents."
        return
    relevant_text, relevant_indices = re_rank_cross_encoders(prompt, docs[0], metas[0])
    if not relevant_text:
        yield "Failed to extract relevant context from the retrieved documents."
        return
    for chunk in call_llm(context=relevant_text, prompt=prompt):
        yield chunk

# ---------------------- Gradio App Layout ----------------------
with gr.Blocks(title="RAG Question Answer (Gradio)") as demo:
    gr.Markdown("# RAG Question Answer App")
    with gr.Tabs():
        with gr.TabItem("File Upload"):
            gr.Markdown("### Upload PDF/DOCX/Image Files for QnA")
            file_input = gr.File(label="Upload Files", file_count="multiple")
            process_btn = gr.Button("Process Files")
            process_output = gr.Textbox(label="Processing Output", lines=10)
        with gr.TabItem("Deletion Options"):
            gr.Markdown("### Delete Files from Vector Store")
            file_dropdown = gr.Dropdown(choices=get_all_file_names(), label="Select file to delete")
            delete_btn = gr.Button("Delete File")
            clear_btn = gr.Button("Clear Entire Vector Store")
            deletion_output = gr.Textbox(label="Deletion Output", lines=5)
        with gr.TabItem("Q&A"):
            gr.Markdown("### Ask a Question")
            prompt_text = gr.Textbox(label="Enter your question", lines=3)
            file_filter = gr.CheckboxGroup(choices=get_all_file_names(), label="Select file(s) to search (optional)")
            qa_btn = gr.Button("Ask Question")
            qa_output = gr.Textbox(label="Answer", lines=10)
        with gr.TabItem("Stored Document Details"):
            gr.Markdown("### Details of Stored Documents")
            stored_details = gr.Textbox(label="Stored Document Details", lines=5)
            refresh_btn = gr.Button("Refresh Stored Document Details")
    
    process_btn.click(fn=process_files, inputs=file_input, outputs=[process_output, file_dropdown, file_filter, stored_details])
    delete_btn.click(fn=delete_file, inputs=file_dropdown, outputs=[deletion_output, file_dropdown, file_filter, stored_details])
    clear_btn.click(fn=clear_vector_store_fn, inputs=None, outputs=[deletion_output, file_dropdown, file_filter, stored_details])
    qa_btn.click(fn=answer_question, inputs=[prompt_text, file_filter], outputs=qa_output)
    refresh_btn.click(fn=get_details, inputs=None, outputs=stored_details)

# Enable queuing to help manage long-running tasks
demo.queue()
demo.launch()
