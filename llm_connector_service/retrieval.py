import os
import logging
from typing import List, Tuple
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAP_DEFS_PATH = os.getenv("SAP_DEFS_PATH", "sap_defs.csv")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

class SAPDefinition:
    def __init__(self, field_name: str, description: str, data_type: str):
        self.field_name = field_name
        self.description = description
        self.data_type = data_type

    def __repr__(self):
        return f"Field: {d.field_name}, Type: {d.data_type}, Description: {d.description}"

class SAPRetrieval:
    def __init__(self, defs_path: str = SAP_DEFS_PATH, model_name: str = EMBEDDING_MODEL_NAME):
        self.defs_path = defs_path
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.definitions: List[SAPDefinition] = []
        self.embeddings: np.ndarray = np.array([])
        self.index = None
        self._load_and_index()

    def _load_and_index(self):
        logger.info(f"Loading SAP definitions from {self.defs_path}")
        df = pd.read_csv(self.defs_path)
        self.definitions = [
            SAPDefinition(row["FieldName"], row["Description"], row["DataType"] + "[" + str(row["Length"]) + "]" )
            for _, row in df.iterrows()
        ]
        descriptions = [str(d) for d in self.definitions]
        logger.info(descriptions)
        logger.info(f"Embedding {len(descriptions)} SAP descriptions...")
        self.embeddings = self.model.encode(descriptions, show_progress_bar=True, convert_to_numpy=True)
        logger.info("Building FAISS index...")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)
        logger.info("FAISS index built and ready.")

    def query(self, text: str, top_k: int = 3) -> List[Tuple[SAPDefinition, float]]:
        logger.info(f"Embedding user query for retrieval: {text}")
        query_emb = self.model.encode([text], convert_to_numpy=True)
        D, I = self.index.search(query_emb, top_k)
        results = [(self.definitions[idx], float(D[0][i])) for i, idx in enumerate(I[0])]
        logger.info(f"Top-{top_k} SAP definitions retrieved.")
        return results

# Singleton instance for FastAPI
sap_retrieval = SAPRetrieval() 