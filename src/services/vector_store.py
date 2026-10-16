import os
import hashlib
from typing import List, Dict, Any, Tuple
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.core.config import settings

class KnowledgeBaseManager:
    def __init__(self):
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(settings.MD5_LOG_PATH), exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self.vector_store = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "? ", "! ", " "]
        )

    def _get_hash(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def is_already_indexed(self, file_hash: str) -> bool:
        if not os.path.exists(settings.MD5_LOG_PATH):
            return False
        with open(settings.MD5_LOG_PATH, "r", encoding="utf-8") as f:
            indexed_hashes = {line.strip() for line in f}
        return file_hash in indexed_hashes

    def _mark_as_indexed(self, file_hash: str):
        with open(settings.MD5_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(file_hash + "\n")

    def index_document(self, raw_chunks: List[Dict[str, Any]], file_hash: str, filename: str) -> Tuple[bool, str]:
        if self.is_already_indexed(file_hash):
            return False, f"Document '{filename}' is already indexed in the knowledge base."

        documents = []
        for item in raw_chunks:
            chunks = self.splitter.split_text(item["content"])
            for idx, c in enumerate(chunks):
                meta = dict(item["metadata"])
                meta["chunk_id"] = idx
                documents.append(Document(page_content=c, metadata=meta))

        if not documents:
            return False, f"No readable text or visual elements found in '{filename}'."

        self.vector_store.add_documents(documents)
        self._mark_as_indexed(file_hash)
        return True, f"Successfully indexed {len(documents)} context chunks from '{filename}'."

    def get_retriever(self, top_k: int = None):
        k = top_k or settings.TOP_K
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def get_stats(self) -> Dict[str, Any]:
        count = self.vector_store._collection.count()
        return {
            "collection_name": settings.COLLECTION_NAME,
            "total_chunks": count,
            "embedding_model": settings.EMBEDDING_MODEL
        }
