import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceEndpoint

from pathlib import Path
import chromadb
from unidecode import unidecode
import re

# Load PDF document and create doc splits
def load_doc(list_file_path, chunk_size=600, chunk_overlap=40):
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages extend(loader.load())
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

# Initialize langchain LLM chain
def initialize_llmchain(temperature, max_tokens, top_k, vector_db):
    llm_model = "mistralai/Mistral-7B-Instruct-v0.2"
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
    response_sources = response["source_documents"]
    response_source1 = response_sources[0].page_content.strip()
    response_source1_page = response_sources[0].metadata["page"] + 1
    new_history = history + [(message, response_answer)]
    return qa_chain, "", new_history, response_source1, response_source1_page

st.title("PDF-based Chatbot")
st.markdown("Upload your PDF and ask any questions about its content.")

uploaded_files = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner('Processing documents...'):
        try:
            list_file_path = [file for file in uploaded_files]
            collection_name = create_collection_name(list_file_path[0].name)
            doc_splits = load_doc(list_file_path)
            vector_db = create_db(doc_splits, collection_name)
            qa_chain = initialize_llmchain(0.7, 1024, 3, vector_db)
            st.success("Ready for questions!")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    history = []
    if "history" not in st.session_state:
        st.session_state["history"] = []

    user_input = st.text_input("Type your question here...")

    if user_input:
        qa_chain, _, st.session_state["history"], response_source1, response_source1_page = conversation(
            qa_chain, user_input, st.session_state["history"])
        st.write(f"Assistant: {st.session_state['history'][-1][1]}")
        st.write(f"Source: {response_source1} (Page {response_source1_page})")
