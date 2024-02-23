from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from constants import openai_key

import os
os.environ['OPENAI_API_KEY'] = openai_key

def read_pdf_text(pdfreader):
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

# Function to split text
def split_text(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    return texts

# Function to perform document search
embeddings = OpenAIEmbeddings()
def perform_document_search(texts, embeddings):
    document_search = FAISS.from_texts(texts, embeddings)
    return document_search

# Function to perform document summary
def summarize_text(raw_text):
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=20)
    chunks = text_splitter.create_documents([raw_text])
    chain1 = load_summarize_chain(llm, chain_type='refine', verbose=True)
    summary = chain1.run(chunks)
    return summary

chain2 = load_qa_chain(OpenAI(),chain_type='stuff')

def run_chain(docs, query):
    return chain2.run(input_documents=docs, question=query)

st.title("Document Summary & Search and QA System")

# File upload section
uploaded_file = st.file_uploader("Upload PDF file", type=['pdf'])

if uploaded_file:
    st.write("PDF file uploaded successfully!")
    # Read PDF and process text
    pdfreader = PdfReader(uploaded_file)
    raw_text = read_pdf_text(pdfreader)
    texts = split_text(raw_text)

    # Perform embeddings
    embeddings = OpenAIEmbeddings()

    #perform document summary
    document_summary = summarize_text(raw_text)
    st.subheader("Summary:")
    st.write(document_summary)

    # Perform document search
    document_search = perform_document_search(texts, embeddings)

    # User query input
    query = st.text_input("Enter your query:")

    if st.button("Search"):
        if query:
            # Perform similarity search
            docs = document_search.similarity_search(query)
                
            # Run chain
            result = run_chain(docs, query)
                
            # Display result
            st.write("Results:")
            st.write(result)
