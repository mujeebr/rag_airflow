from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime
import os
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import openai

# Set OpenAI API key
OPENAI_API_KEY = "your_openai_key"

# Function to update FAISS DB
def update_faiss():
    try:
        pdf_loader = PyPDFDirectoryLoader("/Users/shaikmujeeburrahman/Downloads/ml_airflow/data/pdfs")
        documents = pdf_loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
        chunks = text_splitter.split_documents(documents)

        # Create embeddings and store in FAISS
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_db = FAISS.from_documents(chunks, embeddings)

        faiss_path = "/Users/shaikmujeeburrahman/Downloads/ml_airflow/data/faiss_index"
        os.makedirs(faiss_path, exist_ok=True)
        vector_db.save_local(faiss_path)
        print("FAISS database updated successfully.")
    except Exception as e:
        raise Exception(f"FAISS update failed: {str(e)}")

# Function to log success message
def log_success():
    print("FAISS update completed. Ready for use!")

default_args = {
    "owner": "mujeeb",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

dag = DAG(
    "update_faiss_index",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
)

# Task 1: Update FAISS Index
update_task = PythonOperator(
    task_id="update_faiss",
    python_callable=update_faiss,
    dag=dag,
)

# Task 2: Log success message (runs only if `update_faiss` succeeds)
log_task = PythonOperator(
    task_id="log_success",
    python_callable=log_success,
    dag=dag,
)

# Task 3: Send email on failure (runs only if `update_faiss` fails)
email_task = EmailOperator(
    task_id="send_failure_email",
    to="mujeebr469@gmail.com",
    subject="Airflow DAG Failure: update_faiss_index",
    html_content="The FAISS update task has failed. Please check Airflow logs for details.",
    trigger_rule="one_failed",  # This runs only if at least one upstream task fails
    dag=dag,
)



# Define task dependencies
update_task >> log_task  # log_task runs after update_task
update_task >> email_task  # email_task runs only if update_task fails
