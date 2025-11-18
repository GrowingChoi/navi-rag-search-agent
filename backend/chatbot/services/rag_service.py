"""
검색 → Reranker → LLM 답변 생성까지 전체 RAG workflow
"""

from sentence_transformers import SentenceTransformer
from .rag_search import RAGSearcher
from .answerer import AnswerGenerator
from .filters import is_form_related_query
from .constants import RERANKER_TOP_K


class RAGService:

    def __init__(self):
        self.embed_model = SentenceTransformer("upstage/koe5")
        self.searcher = RAGSearcher()
        self.answerer = AnswerGenerator()

    def embed(self, text):
        return self.embed_model.encode(text).tolist()

    def search_documents(self, question_vector):

        results = self.searcher.search(query_vector=question_vector)
        return results[:RERANKER_TOP_K]

    def generate_answer(self, question, docs):

        context = "\n\n".join(
            [f"[{i+1}] {doc.payload['text']}" for i, doc in enumerate(docs)]
        )

        return self.answerer.generate(question, context)

    def run(self, question):

        q_vec = self.embed(question)

        docs = self.search_documents(q_vec)

        answer = self.generate_answer(question, docs)

        return answer, [d.payload for d in docs]
