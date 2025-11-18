import os
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from django.conf import settings

def _read_pdf_texts(pdf_path: Path) -> List[Dict]:
    """PDF를 페이지 단위로 텍스트 추출"""
    reader = PdfReader(str(pdf_path))
    results = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            results.append({"page": i, "text": text})
    return results

def _chunk(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        start += chunk_size - overlap
        if start < 0 or start >= n:
            break
    return [c.strip() for c in chunks if c.strip()]

def ensure_collection(client: QdrantClient, name: str, vector_size: int):
    exists = False
    try:
        info = client.get_collection(name)
        exists = True if info else False
    except Exception:
        exists = False

    if not exists:
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

def build_embeddings_model(model_name: str = "nlpai-lab/KoE5"):
    return SentenceTransformer(model_name)

def index_directory(
    dir_path: str,
    collection_name: str = None,
    vector_size: int = None,
):
    collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
    vector_size = vector_size or settings.QDRANT_VECTOR_SIZE

    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    ensure_collection(client, collection_name, vector_size)

    model = build_embeddings_model()
    base = Path(dir_path)

    pdf_files = sorted(list(base.glob("**/*.pdf")))
    if not pdf_files:
        print(f"[indexer] No PDFs found under: {dir_path}")
        return

    point_id = 1
    for pdf in tqdm(pdf_files, desc="Indexing PDFs"):
        pages = _read_pdf_texts(pdf)
        for item in pages:
            for chunk in _chunk(item["text"]):
                vec = model.encode([chunk])[0].tolist()
                client.upsert(
                    collection_name=collection_name,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=vec,
                            payload={
                                "source": pdf.name,
                                "path": str(pdf),
                                "page": item["page"],
                                "text": chunk,
                            },
                        )
                    ],
                )
                point_id += 1

    print(f"[indexer] Done. Upserted up to point id: {point_id-1}") 