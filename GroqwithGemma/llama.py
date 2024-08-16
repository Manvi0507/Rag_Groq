import streamlit as st
import os
import requests
import urllib
import subprocess
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the GROQ and Google API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Medical Document Q&A")

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
"""
)

# Clone the GitHub repository (run only once)
repo_url = "https://github.com/Manvi0507/Rag_Groq/tree/7f8221a6eea692eda77dc38deef9b84a300b1cb6/GroqwithGemma.git"
repo_dir = "./med_data"

if not os.path.exists(repo_dir):
    subprocess.run(["git", "clone", repo_url, repo_dir])

# Set document directory path
document_directory = os.path.join(repo_dir, "docs")

# Function to handle vector embedding
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader(document_directory)  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        
        # Ensure documents are loaded
        if not st.session_state.docs:
            st.write("No documents were loaded. Please check the document directory.")
            return

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
        
        # Ensure documents are split into chunks
        if not st.session_state.final_documents:
            st.write("No document chunks were created. Please check the documents.")
            return
        
        # Try to create FAISS vector embeddings
        try:
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        except Exception as e:
            st.write(f"An error occurred during vector embedding: {e}")
            return

# Run the vector embedding function when the button is clicked
if st.button("Documents Embedding"):
    vector_embedding()
    if "vectors" in st.session_state:
        st.write("Vector Store DB Is Ready")
    else:
        st.write("Failed to initialize the vector store.")

# Handle the Ask Question button with error checking
if st.button("Ask Question"):
    if "vectors" not in st.session_state:
        st.write("Please embed the documents first.")
    else:
        if prompt1:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            st.write("Response time: ", time.process_time() - start)
            st.write(response['answer'])

            # Display relevant document chunks in an expander
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
