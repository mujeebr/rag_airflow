from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import openai

# Set your OpenAI API Key

# # Define function to update FAISS index
# def update_faiss():
#     pdf_loader = PyPDFDirectoryLoader("data/pdfs")
#     documents = pdf_loader.load()
    
#     # Split documents into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
#     chunks = text_splitter.split_documents(documents)

#     # Create embeddings and store in FAISS
#     embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#     vector_db = FAISS.from_documents(chunks, embeddings)

#     # Save FAISS index
#     faiss_path = "data/faiss_index"
#     os.makedirs(faiss_path, exist_ok=True)
#     vector_db.save_local(faiss_path)

# # Define Airflow DAG
# default_args = {
#     "owner": "airflow",
#     "start_date": datetime(2024, 1, 1),
#     "retries": 1,
# }

# dag = DAG(
#     "update_faiss_index",
#     default_args=default_args,
#     schedule_interval="@daily",  # Runs every day
#     catchup=False,
# )

# # Define Airflow task
# update_task = PythonOperator(
#     task_id="update_faiss",
#     python_callable=update_faiss,
#     dag=dag,
# )

# update_task


from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Define function to update FAISS index
def update_faiss():
    pdf_loader = PyPDFDirectoryLoader("/Users/shaikmujeeburrahman/Downloads/ml_airflow/data/pdfs")  # Use absolute path
    documents = pdf_loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings and store in FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_db = FAISS.from_documents(chunks, embeddings)

    # Save FAISS index
    faiss_path = "/Users/shaikmujeeburrahman/Downloads/ml_airflow/data/faiss_index"
    os.makedirs(faiss_path, exist_ok=True)
    vector_db.save_local(faiss_path)

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

dag = DAG(
    "update_faiss_index",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
)

update_task = PythonOperator(
    task_id="update_faiss",
    python_callable=update_faiss,
    dag=dag,
)

update_task
