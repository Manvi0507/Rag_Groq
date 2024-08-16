import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pdfplumber

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Function to extract text from multiple PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to handle text chunks for Vector Store
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to get Vector Store using FAISS with Google embeddings
def get_vector_store(text_chunks):
    # Use Google's Generative AI Embeddings for embedding the chunks
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# Function to initialize conversational chain with Groq Llama model
def get_conversational_chain(vector_store):
    prompt = """
    You are an AI assistant designed to summarize and extract information from documents. Follow these instructions carefully:

    - Summarize the provided document in no more than 200 words.
    - Ensure the summary is concise and captures the main points.
    - Use clear headings and bullet points to organize the information when needed.
    - If a table format is suitable for presenting the information, format the response accordingly.
    - Base your response strictly on the content of the provided document.
    - Incorporate the conversation history to provide contextually relevant and coherent answers.

    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}

    Answer:
    """

    # Groq model for conversational tasks
    model = ChatGroq(api_key=groq_api_key, model_name="Llama3-8b-8192")  

    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=["context", "question", "history"]
    )

    memory = st.session_state.get("memory", ConversationBufferMemory(
        memory_key="history",
        input_key="question"
    ))

    if vector_store is None:
        st.error("Vector Store is not initialized. Please upload and process your documents first.")
        return None

    retrieval_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type='stuff',
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template, "memory": memory}
    )

    st.session_state.memory = memory

    return retrieval_chain

# Function to handle user input and response
def handle_user_input(user_question):
    vector_store = st.session_state.get("vector_store", None)
    if not vector_store:
        st.error("Please upload and process PDF files first.")
        return

    chain = get_conversational_chain(vector_store)
    if chain is None:
        return

    response = chain({"query": user_question}, return_only_outputs=True)
    st.session_state.response = response.get("result", "No valid response received.")
    st.write("Reply: ", st.session_state.response)

# Main function to run Streamlit app
def main():
    st.set_page_config(page_title="Chat with Documents using Groq & Google")
    st.header("QA with Your Multiple Documents 💁")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "response" not in st.session_state:
        st.session_state.response = ""

    user_question = st.text_input("Ask a Question from the PDF Files", value=st.session_state.response)

    if st.button("Ask Question"):
        handle_user_input(user_question)

    if st.button("Clear Response"):
        st.session_state.response = ""
        st.write("Response cleared.")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
        if pdf_docs and st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Process all uploaded PDFs
                raw_text = get_pdf_text(pdf_docs)
                
                # Split the text into chunks
                text_chunks = get_text_chunks(raw_text)
                
                # Create a vector store from the text chunks
                st.session_state.vector_store = get_vector_store(text_chunks)
                st.success("Documents processed successfully!")

if __name__ == "__main__":
    main()
