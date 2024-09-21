from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import SVMRetriever
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Initialize embeddings using Hugging Face
def get_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

# Function to create FAISS vector store from provided data
def create_vector_store(questions, embeddings):
    texts = [q['text'] for q in questions]
    # Create a FAISS index
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store

# Initialize Hugging Face LLM for LangChain
def get_llm(api_key, model_name="gpt-neo-2.7B"):
    return HuggingFaceHub(api_key=api_key, model_name=model_name)

# Function to load pre-trained LLM with embedding-based similarity search
def create_retrieval_chain(questions, api_key, model_name="gpt-neo-2.7B"):
    # Initialize embeddings
    embeddings = get_embeddings()

    # Create vector store using FAISS for the question bank
    vector_store = create_vector_store(questions, embeddings)

    # Initialize the pre-trained LLM
    llm = get_llm(api_key, model_name)

    # Create a retriever with similarity search using FAISS
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Create the RetrievalQA chain
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return retrieval_qa

# Perform RAG (Retrieval-Augmented Generation) using LangChain
def rag_processing(questions, api_key, model_name="gpt-neo-2.7B", input_text=""):
    # Create the retrieval chain
    retrieval_qa = create_retrieval_chain(questions, api_key, model_name)

    # Query the chain using the input text
    response = retrieval_qa.run(input_text)

    return response

# Function to run similarity search based on a student's answer
def similarity_search(questions, answer, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    # Initialize embeddings
    embeddings = get_embeddings(model_name)

    # Create vector store for the question bank
    vector_store = create_vector_store(questions, embeddings)

    # Perform a similarity search
    docs = vector_store.similarity_search(answer, k=3)  # Retrieve top 3 closest matches
    return docs

# Evaluate student's answer using LangChain's RetrievalQA
def evaluate_answer(questions, student_answer, api_key, model_name="gpt-neo-2.7B"):
    # Process the answer using RAG to generate feedback
    response = rag_processing(questions, api_key, model_name, input_text=student_answer)

    # Return the generated evaluation
    return response
