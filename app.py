# import streamlit as st
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.document_loaders import PyPDFDirectoryLoader
# from langchain.embeddings import OpenAIEmbeddings

# # Streamlit app title
# st.title("Full Stack Academy Info Finder")

# # Sidebar
# st.sidebar.header("Configuration")
# api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# # Input Section
# st.header("Search Full Stack Academy Info")
# question = st.text_input("Enter your question:")
# # api_key = 


# # Load data and set up the chain when user provides API Key
# if api_key:
#     try:
#         # Load URLs and process data
#         # URLs = ["https://fullstackacademy.in/"]
#         # loaders = UnstructuredURLLoader(urls=URLs)
#         loaders=PyPDFDirectoryLoader("pdfs")
#         data = loaders.load()

#         # Split text into chunks
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
#         chunks = text_splitter.split_documents(data)

#         # Create embeddings and vector store
#         # embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
#         vectordatabase = FAISS.from_documents(chunks, embedding_model)

#         # Initialize LLM
#         llm = OpenAI(api_key=api_key)

#         # Define prompt template
#         template = """Use the context to provide a concise answer. If you don't know, just say 'I don't know.'
#         {context}
#         Question: {question}
#         Helpful Answer:"""
#         QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

#         # Create retrieval chain
#         chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=vectordatabase.as_retriever(),
#             chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
#         )

#         # Run query if question is entered
#         if question:
#             with st.spinner("Searching..."):
#                 answer = chain.run(question)
#             st.subheader("Answer:")
#             st.write(answer)
#     except Exception as e:
#         st.error(f"Error: {str(e)}")
# else:
#     st.warning("Please enter your OpenAI API key in the sidebar.")

# pkill -f "airflow scheduler"
# pkill -f "airflow webserver"
import streamlit as st
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Streamlit UI
st.title("Full Stack Academy Info Finder")

# Sidebar for API key
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Input Section
st.header("Ask a Question")
question = st.text_input("Enter your question:")

if api_key:
    try:
        # Load FAISS index
        faiss_path = "data/faiss_index"
        if not os.path.exists(faiss_path):
            st.error("FAISS index not found! Run the Airflow DAG to create embeddings.")
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            vectordatabase = FAISS.load_local(faiss_path, embeddings,allow_dangerous_deserialization=True)

            # Initialize LLM
            llm = OpenAI(api_key=api_key)

            # Define prompt
            template = """Use the context to provide a concise answer. If you don't know, just say 'I don't know.'
            {context}
            Question: {question}
            Helpful Answer:"""
            QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

            # Retrieval chain
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectordatabase.as_retriever(),
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            )

            # Get answer
            if question:
                with st.spinner("Searching..."):
                    answer = chain.run(question)
                st.subheader("Answer:")
                st.write(answer)

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.warning("Please enter your OpenAI API key.")
