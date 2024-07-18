import streamlit as st
import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.llms import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory

from pathlib import Path
import chromadb
from unidecode import unidecode
import re

# Load PDF document and create doc splits
def load_doc(list_file_path, chunk_size=600, chunk_overlap=40):
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
def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db):
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

    uploaded_files = st.file_uploader("Upload your PDF documents (single or multiple)", accept_multiple_files=True, type=["pdf"])

    if uploaded_files:
        with st.spinner("Processing documents and initializing..."):
            vector_db, collection_name = initialize_database(uploaded_files)
            qa_chain = initialize_llmchain("mistralai/Mistral-7B-Instruct-v0.2", 0.7, 1024, 3, vector_db)
            st.session_state.qa_chain = qa_chain
            st.session_state.chat_history = []
            st.success("Initialization complete. You can start asking questions now.")

    if "qa_chain" in st.session_state:
        message = st.text_input("Type message (e.g. 'What is this document about?')")
        if st.button("Submit message"):
            qa_chain, chat_history, source1, source1_page, source2, source2_page, source3, source3_page = conversation(st.session_state.qa_chain, message, st.session_state.chat_history)
            st.session_state.chat_history = chat_history
            st.write("Chatbot:", chat_history)
            st.write(f"Reference 1 (Page {source1_page}): {source1}")
            st.write(f"Reference 2 (Page {source2_page}): {source2}")
            st.write(f"Reference 3 (Page {source3_page}): {source3}")

def initialize_database(uploaded_files):
    list_file_path = [x.name for x in uploaded_files if x is not None]
    collection_name = create_collection_name(list_file_path[0])
    doc_splits = load_doc(list_file_path)
    vector_db = create_db(doc_splits, collection_name)
    return vector_db, collection_name

if __name__ == "__main__":
    main()
