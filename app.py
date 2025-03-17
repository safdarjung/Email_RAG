import os
import tempfile
import concurrent.futures
import chromadb
import ollama
import streamlit as st
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import PdfFormatOption
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

# ---------------------- System Prompt ----------------------
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

# ---------------------- Custom Text Splitter ----------------------
def get_text_splitter(method: str) -> RecursiveCharacterTextSplitter:
    """
    Returns a RecursiveCharacterTextSplitter configured based on the selected chunking method.
    """
    if method == "Word":
        return RecursiveCharacterTextSplitter(
            chunk_size=250,  # Approximate character count (~50 words)
            chunk_overlap=50,
            separators=[" ", ""],
        )
    elif method == "Sentence":
        return RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separators=[". ", " ", ""],
        )
    elif method == "Paragraph":
        return RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
    elif method == "Topic":
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
    elif method == "Pages Batch":
        return RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
    else:
        # Default to Paragraph splitting
        return RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )

# ---------------------- Document Processing ----------------------
def process_file(uploaded_file: UploadedFile, chunking_method: str = "Paragraph") -> list[Document]:
    """
    Processes an uploaded file using Docling and converts it into text chunks using the selected chunking method.
    """
    extension = os.path.splitext(uploaded_file.name)[1].lower()
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=extension, delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    # Configure pipeline options for Docling
    pipeline_options = PdfPipelineOptions()
    pipeline_options.ocr_options = EasyOcrOptions()

    try:
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        result = converter.convert(temp_file.name)
        docling_doc = result.document

        # Extract text from the document
        text = docling_doc.export_to_markdown()

        # Create a Document object
        doc = Document(page_content=text, metadata={"source": uploaded_file.name})

        # Split the text if chunking method is not "Page"
        if chunking_method != "Page":
            text_splitter = get_text_splitter(chunking_method)
            splits = text_splitter.split_documents([doc])
        else:
            splits = [doc]

        return splits

    except Exception as e:
        st.error(f"Failed to process {uploaded_file.name}: {e}")
        return []

    finally:
        os.unlink(temp_file.name)

# ---------------------- ChromaDB Setup ----------------------
def get_vector_collection() -> chromadb.Collection:
    """
    Initializes a persistent ChromaDB vector store with an Ollama embedding function.
    """
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
def add_to_vector_collection(all_splits: list[Document], file_name: str):
    """
    Upserts all text chunks from a file to the vector store in a single batch.
    Each chunk’s metadata is augmented with a 'source' key containing the file name.
    """
    collection = get_vector_collection()
    documents = []
    metadatas = []
    ids = []
    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadata = split.metadata if split.metadata else {}
        metadata["source"] = file_name
        metadatas.append(metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success(f"Data from {file_name} added to the vector store!")

# ---------------------- Deletion Functions ----------------------
def delete_file_from_vector_store(file_name: str):
    """
    Deletes all documents in the vector store that have metadata 'source' equal to the given file name.
    """
    collection = get_vector_collection()
    collection.delete(where={"source": file_name})
    st.success(f"Data from {file_name} deleted from the vector store!")

def clear_vector_store():
    """
    Clears the entire vector store by retrieving all document IDs and deleting them.
    """
    collection = get_vector_collection()
    all_data = collection.get()
    if "ids" in all_data and all_data["ids"]:
        collection.delete(ids=all_data["ids"])
        st.success("Vector store cleared!")
    else:
        st.info("Vector store is already empty.")

# ---------------------- Query & LLM Calls ----------------------
def query_collection(prompt: str, n_results: int = 10):
    """
    Queries the vector collection with the provided prompt.
    """
    collection = get_vector_collection()
    return collection.query(query_texts=[prompt], n_results=n_results)

def call_llm(context: str, prompt: str):
    """
    Streams a response from the LLM via Ollama given the context and the user's question.
    """
    response = ollama.chat(
        model="deepseek-r1:8b",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}, Question: {prompt}"},
        ],
    )
    for chunk in response:
        if not chunk.get("done", True):
            yield chunk["message"]["content"]
        else:
            break

def re_rank_cross_encoders(query: str, documents: list[str]) -> tuple[str, list[int]]:
    """
    Uses a cross-encoder to re-rank retrieved document chunks based on relevance.
    Returns concatenated text from the top 3 documents and their indices.
    """
    if not documents:
        return "", []
    relevant_text = ""
    relevant_text_ids = []
    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(query, documents, top_k=3)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])
    return relevant_text, relevant_text_ids

# ---------------------- Parallel Processing Function ----------------------
def process_and_upsert_file(uploaded_file: UploadedFile, chunking_method: str) -> tuple[str, bool]:
    """
    Processes and upserts a single file.
    Returns a tuple of the normalized file name and a success flag.
    """
    normalized_name = uploaded_file.name.translate(
        str.maketrans({"-": "_", ".": "_", " ": "_"})
    )
    splits = process_file(uploaded_file, chunking_method)
    if splits:
        add_to_vector_collection(splits, normalized_name)
        return normalized_name, True
    else:
        return normalized_name, False

# ---------------------- Streamlit UI ----------------------
if __name__ == "__main__":
    st.set_page_config(page_title="RAG Question Answer", layout="wide")
    
    # Sidebar: File upload, chunking method selection, processing, and deletion.
    with st.sidebar:
        st.header("File Processing")
        uploaded_files = st.file_uploader(
            "**📑 Upload PDF/DOC/Image files for QnA**",
            type=["pdf", "docx", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
        chunking_method = st.selectbox(
            "Select Chunking Method",
            options=["Word", "Sentence", "Paragraph", "Topic", "Page", "Pages Batch"],
            index=2  # Default to "Paragraph"
        )
        process = st.button("⚡️ Process Files")
        
        if uploaded_files and process:
            total_files = len(uploaded_files)
            overall_progress = st.progress(0)
            st.write("### Processing Files in Parallel")
            results = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(process_and_upsert_file, file, chunking_method)
                    for file in uploaded_files
                ]
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()  # (normalized_name, success)
                    results.append(result)
                    completed += 1
                    overall_progress.progress(completed / total_files)
                    st.write(f"Processed: **{result[0]}** - {'Success' if result[1] else 'Failed'}")
            st.success("All files processed successfully!")
        
        st.header("Deletion Options")
        file_to_delete = st.text_input("Enter file name to delete (normalized name)")
        if st.button("🗑️ Delete File"):
            if file_to_delete:
                delete_file_from_vector_store(file_to_delete)
            else:
                st.warning("Please enter a file name to delete.")
        if st.button("🗑️ Clear Entire Vector Store"):
            clear_vector_store()
    
    # Main: Question and Answer Interface
    st.header("🗣️ RAG Question Answer")
    prompt = st.text_area("**Ask a question related to your uploaded documents:**")
    ask = st.button("🔥 Ask Question")
    
    if ask and prompt:
        results = query_collection(prompt)
        docs = results.get("documents", [])
        
        if docs and len(docs) > 0 and docs[0]:
            context_docs = docs[0]
        else:
            st.warning("No relevant documents found. Please try a different query.")
            st.stop()
        
        relevant_text, relevant_text_ids = re_rank_cross_encoders(prompt, context_docs)
        if not relevant_text:
            st.warning("Failed to extract relevant context from the retrieved documents.")
        else:
            response_stream = call_llm(context=relevant_text, prompt=prompt)
            st.write_stream(response_stream)
        
        with st.expander("See Retrieved Documents"):
            st.write(results)
        with st.expander("See Most Relevant Document IDs"):
            st.write(relevant_text_ids)
            st.write(relevant_text)