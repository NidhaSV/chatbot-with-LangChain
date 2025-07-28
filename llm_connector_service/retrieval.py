from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic

from langchain.prompts import PromptTemplate
import os
import pandas as pd
import pickle

SAP_DEFS_PATH = os.getenv("SAP_DEFS_PATH", "sap_defs.csv")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DB_FAISS_PATH = "faiss_index"

# STEP 1: Load documents from CSV
def load_sap_definitions_from_csv(csv_path: str) -> list[Document]:
    df = pd.read_csv(csv_path)
    documents = []

    for _, row in df.iterrows():
        metadata = {
            "field_name": row["FieldName"],
            "data_type": f"{row['DataType']}[{row['Length']}]"
        }
        text = f"Field: {metadata['field_name']}, Type: {metadata['data_type']}, Description: {row['Description']}"
        documents.append(Document(page_content=text, metadata=metadata))
    
    return documents

# STEP 2: Embeddings
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# STEP 3: Load or Create FAISS vector store
if os.path.exists(f"{DB_FAISS_PATH}/index.faiss"):
    with open(f"{DB_FAISS_PATH}/faiss_store.pkl", "rb") as f:
        vectorstore = pickle.load(f)
else:
    print("Creating FAISS vector store...")
    documents = load_sap_definitions_from_csv(SAP_DEFS_PATH)
    vectorstore = FAISS.from_documents(documents, embedding_model)
    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    vectorstore.save_local(DB_FAISS_PATH)
    with open(f"{DB_FAISS_PATH}/faiss_store.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

# STEP 4: Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# STEP 5: Prompt Template with strict instruction
template = """IMPORTANT: Do not guess or make up expansions for abbreviations or SAP terms.
Only use the definitions provided in the context. If you're unsure, say 'Definition not found.'

Relevant SAP Definitions:
{context}

User Query:
{question}
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# STEP 6: LLM
llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)

# STEP 7: RAG Chain (retrieval + response)
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)
