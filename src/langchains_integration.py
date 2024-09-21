from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()


os.environ["HUGGING_FACE_API_KEY"] = os.getenv("HUGGING_FACE_API_KEY")

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def create_vector_store(text_chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def rag_processing(questions, api_key, input_text=""):

    vector_store = create_vector_store(questions)
    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.7})

    response = llm(input_text)
    return response
