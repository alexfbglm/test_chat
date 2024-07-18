import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceEndpoint

from pathlib import Path
import chromadb
from unidecode import unidecode

from transformers import AutoTokenizer
import transformers
import torch
import tqdm 
import accelerate
import re

# LLM List
list_llm = ["mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.1", \
    "google/gemma-7b-it","google/gemma-2b-it", \
    "HuggingFaceH4/zephyr-7b-beta", "HuggingFaceH4/zephyr-7b-gemma-v0.1", \
    "meta-llama/Llama-2-7b-chat-hf", "microsoft/phi-2", \
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "mosaicml/mpt-7b-instruct", "tiiuae/falcon-7b-instruct", \
    "google/flan-t5-xxl"
]
list_llm_simple = [os.path.basename(llm) for llm in list_llm]

# Load PDF document and create doc splits
def load_doc(list_file_path, chunk_size, chunk_overlap):
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits

# Create vector database
def create_db(splits, collection_name):
    embedding = HuggingFaceEmbeddings()
    new_client = chromadb.EphemeralClient()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        client=new_client,
        collection_name=collection_name,
    )
    return vectordb

# Load vector database
def load_db():
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma(
        embedding_function=embedding)
    return vectordb

# Initialize langchain LLM chain
def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db):
    st.write("Initializing HF Hub...")
    if llm_model == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        llm = HuggingFaceEndpoint(
            repo_id=llm_model,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_k=top_k,
            load_in_8bit=True,
        )
    elif llm_model in ["HuggingFaceH4/zephyr-7b-gemma-v0.1", "mosaicml/mpt-7b-instruct"]:
        st.error("LLM model is too large to be loaded automatically on free inference endpoint")
    elif llm_model == "microsoft/phi-2":
        llm = HuggingFaceEndpoint(
            repo_id=llm_model,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_k=top_k,
            trust_remote_code=True,
            torch_dtype="auto",
        )
    elif llm_model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        llm = HuggingFaceEndpoint(
            repo_id=llm_model,
            temperature=temperature,
            max_new_tokens=250,
            top_k=top_k,
        )
    elif llm_model == "meta-llama/Llama-2-7b-chat-hf":
        st.error("Llama-2-7b-chat-hf model requires a Pro subscription...")
    else:
        llm = HuggingFaceEndpoint(
            repo_id=llm_model,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_k=top_k,
        )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    retriever = vector_db.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return qa_chain

# Generate collection name for vector database
def create_collection_name(filepath):
    collection_name = Path(filepath).stem
    collection_name = collection_name.replace(" ", "-")
    collection_name = unidecode(collection_name)
    collection_name = re.sub('[^A-Za-z0-9]+', '-', collection_name)
    collection_name = collection_name[:50]
    if len(collection_name) < 3:
        collection_name = collection_name + 'xyz'
    if not collection_name[0].isalnum():
        collection_name = 'A' + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + 'Z'
    return collection_name

# Initialize database
def initialize_database(list_file_obj, chunk_size, chunk_overlap):
    list_file_path = [x.name for x in list_file_obj if x is not None]
    collection_name = create_collection_name(list_file_path[0])
    doc_splits = load_doc(list_file_path, chunk_size, chunk_overlap)
    vector_db = create_db(doc_splits, collection_name)
    return vector_db, collection_name

def format_chat_history(message, chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history

def conversation(qa_chain, message, history):
    formatted_chat_history = format_chat_history(message, history)
    response = qa_chain({"question": message, "chat_history": formatted_chat_history})
    response_answer = response["answer"]
    if "Helpful Answer:" in response_answer:
        response_answer = response_answer.split("Helpful Answer:")[-1]
    response_sources = response["source_documents"]
    response_source1 = response_sources[0].page_content.strip()
    response_source2 = response_sources[1].page_content.strip()
    response_source3 = response_sources[2].page_content.strip()
    response_source1_page = response_sources[0].metadata["page"] + 1
    response_source2_page = response_sources[1].metadata["page"] + 1
    response_source3_page = response_sources[2].metadata["page"] + 1
    new_history = history + [(message, response_answer)]
    return qa_chain, new_history, response_source1, response_source1_page, response_source2, response_source2_page, response_source3, response_source3_page

def main():
    st.title("PDF-based Chatbot")
    st.write("Ask any questions about your PDF documents")

    st.sidebar.header("Step 1 - Upload PDF")
    uploaded_files = st.sidebar.file_uploader("Upload your PDF documents (single or multiple)", accept_multiple_files=True, type=["pdf"])

    st.sidebar.header("Step 2 - Process document")
    db_type = st.sidebar.radio("Vector database type", ["ChromaDB"], index=0)
    chunk_size = st.sidebar.slider("Chunk size", 100, 1000, 600, 20)
    chunk_overlap = st.sidebar.slider("Chunk overlap", 10, 200, 40, 10)
    
    if st.sidebar.button("Generate vector database"):
        with st.spinner("Initializing vector database..."):
            vector_db, collection_name = initialize_database(uploaded_files, chunk_size, chunk_overlap)
            st.session_state.vector_db = vector_db
            st.session_state.collection_name = collection_name
            st.success("Vector database initialized")

    st.sidebar.header("Step 3 - Initialize QA chain")
    llm_option = st.sidebar.selectbox("LLM models", list_llm_simple)
    temperature = st.sidebar.slider("Temperature", 0.01, 1.0, 0.7, 0.1)
    max_tokens = st.sidebar.slider("Max Tokens", 224, 4096, 1024, 32)
    top_k = st.sidebar.slider("top-k samples", 1, 10, 3, 1)
    
    if st.sidebar.button("Initialize Question Answering chain"):
        with st.spinner("Initializing QA chain..."):
            qa_chain = initialize_llmchain(list_llm[llm_option], temperature, max_tokens, top_k, st.session_state.vector_db)
            st.session_state.qa_chain = qa_chain
            st.success("QA chain initialized")

    st.header("Step 4 - Chatbot")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "qa_chain" in st.session_state:
        message = st.text_input("Type message (e.g. 'What is this document about?')")
        if st.button("Submit message"):
            qa_chain, chat_history, source1, source1_page, source2, source2_page, source3, source3_page = conversation(st.session_state.qa_chain, message, st.session_state.chat_history)
            st.session_state.chat_history = chat_history
            st.write("Chatbot:", chat_history)
            st.write(f"Reference 1 (Page {source1_page}): {source1}")
            st.write(f"Reference 2 (Page {source2_page}): {source2}")
            st.write(f"Reference 3 (Page {source3_page}): {source3}")

if __name__ == "__main__":
    main()
