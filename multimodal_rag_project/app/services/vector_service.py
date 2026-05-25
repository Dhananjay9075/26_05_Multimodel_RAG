# title="app/services/vector_service.py"
import os
import uuid
import lancedb
import numpy as np
from typing import List, Dict, Any
from app.config import settings

_text_embed_model = None
_lancedb_connection = None
TABLE_NAME = "chat_session"

def get_text_embedder():
    global _text_embed_model
    if _text_embed_model is None:
        from sentence_transformers import SentenceTransformer
        # Switched to MiniLM: Uses ~80MB of RAM instead of 3.2GB!
        _text_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _text_embed_model

def get_lancedb_client():
    global _lancedb_connection
    if _lancedb_connection is None:
        db_path = os.path.join(os.getcwd(), "local_vector_db")
        os.makedirs(db_path, exist_ok=True)
        _lancedb_connection = lancedb.connect(db_path)
    return _lancedb_connection

class VectorDBService:
    @staticmethod
    def reset_vector_space():
        """Completely deletes the local vector database table for a fresh start."""
        db = get_lancedb_client()
        if TABLE_NAME in db.table_names():
            db.drop_table(TABLE_NAME)
            print("🧹 Local LanceDB vector space wiped clean for new chat/upload.")

    @staticmethod
    def index_chunks(chunks: List[Dict[str, Any]]):
        """Generates dense vector representations and publishes to local LanceDB."""
        if not chunks:
            return
            
        try:
            import gc
            import torch
            db = get_lancedb_client()
            embedder = get_text_embedder()
            
            data = []
            print(f"🧩 Embedding {len(chunks)} chunks individually to save RAM...")
            
            for chunk in chunks:
                # 1. Encode ONE chunk at a time to prevent 3.2GB memory spikes
                emb = embedder.encode(
                    [chunk["content"]], 
                    convert_to_numpy=True, 
                    batch_size=1
                ).tolist()[0]
                
                data.append({
                    "id": chunk["id"],
                    "vector": emb,
                    "content": chunk["content"],
                    "source_file": chunk["metadata"]["source_file"],
                    "page_index": chunk["metadata"]["page_index"]
                })
                
                # 2. Force memory release immediately after each chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Append if table exists, create if it doesn't
            if TABLE_NAME in db.table_names():
                table = db.open_table(TABLE_NAME)
                table.add(data)
            else:
                db.create_table(TABLE_NAME, data=data)
                
            print(f"✨ Successfully saved {len(chunks)} vectors to local LanceDB!")
                
        except Exception as ex:
            print(f"[LanceDB Storage Error]: {ex}")

    @staticmethod
    def query_vector_space(query_text: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Executes query embedding generation and local retrieval."""
        try:
            db = get_lancedb_client()
            
            # If the database is completely empty/new, return nothing
            if TABLE_NAME not in db.table_names():
                return []
                
            embedder = get_text_embedder()
            query_vector = embedder.encode(query_text, convert_to_numpy=True).tolist()
            
            table = db.open_table(TABLE_NAME)
            # Search the local file structure
            results = table.search(query_vector).limit(top_k).to_list()
            
            retrieved_matches = []
            for match in results:
                retrieved_matches.append({
                    "content": match["content"],
                    "source": match["source_file"],
                    "page": match["page_index"],
                    "score": match.get("_distance", 0.0)
                })
            return retrieved_matches
        except Exception as e:
            print(f"Query Error: {e}")
            return []