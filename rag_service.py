from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
import google.generativeai as genai
from settings import settings
from database import get_qdrant_client
import uuid

class RAGService:
    def __init__(self):
        # OpenAI SDK for embeddings
        self.openai_client = OpenAI(
            api_key=settings.GEMINI_API_KEY,
            base_url=settings.GEMINI_BASE_URL
        )
        # Native Google SDK for chat
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel(settings.CHAT_MODEL)
        
        self.qdrant_client = get_qdrant_client()
        self.collection_name = "documents"

    def generate_embedding(self, text: str) -> list[float]:
        response = self.openai_client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding

    def upsert_document(self, text: str):
        embedding = self.generate_embedding(text)
        point_id = str(uuid.uuid4())
        
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"text": text}
                )
            ]
        )
        return point_id

    def search(self, query: str, limit: int = 3) -> list[str]:
        query_embedding = self.generate_embedding(query)
        
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit
        )
        
        return [hit.payload["text"] for hit in search_result.points]

    def generate_answer(self, query: str) -> str:
        context_docs = self.search(query)
        
        if not context_docs:
            return "I don't have enough information to answer that question."
        
        context_str = "\n\n".join(context_docs)
        prompt = f"""You are a helpful assistant. Use the provided context to answer the user's question.

Context:
{context_str}

Question: {query}

Answer:"""
        
        response = self.gemini_model.generate_content(prompt)
        return response.text

rag_service = RAGService()
