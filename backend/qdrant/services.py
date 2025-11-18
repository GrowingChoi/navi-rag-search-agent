import os
import re
import uuid
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import pypdf
import logging
from django.conf import settings

logger = logging.getLogger(__name__)


class QdrantService:
    """Qdrant 벡터 데이터베이스 서비스 (고도화 버전)"""
    
    def __init__(self, host: str = None, port: int = None):
        # Django 설정에서 기본값 가져오기
        self.host = host or getattr(settings, 'QDRANT_HOST', 'localhost')
        self.port = port or getattr(settings, 'QDRANT_PORT', 6333)
        self.collection_name = getattr(settings, 'QDRANT_COLLECTION_NAME', 'kisa_documents')
        self.vector_size = getattr(settings, 'QDRANT_VECTOR_SIZE', 1024)  # KoE5 모델용
        
        # 클라이언트 및 모델 초기화
        self.client = QdrantClient(host=self.host, port=self.port)
        self.embedding_model = HuggingFaceEmbeddings(model_name="nlpai-lab/KoE5")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        # 카테고리 키워드 정의
        self.category_keywords = {
            "보안 규정": ["보안", "정보보호"],
            "복리후생 규정": ["복리후생", "급여", "복지", "휴가"],
            "복무 규정": ["복무", "출근", "근무", "지각"],
            "안전보건관리 규정": ["안전", "보건", "산재", "재해"],
            "여비 규정": ["여비", "출장비", "교통비"],
            "인사 규정": ["인사", "채용", "평가", "승진"],
            "인사팀 업무 가이드": ["인사팀", "인사담당"],
            "전산팀 업무 가이드": ["전산팀", "IT", "서버"],
            "회계 규정": ["회계", "전표", "지출"],
            "회계팀 업무 가이드": ["회계팀", "세무", "회계업무"]
        }
        
        # 카테고리 분류 프롬프트
        self.category_prompt = PromptTemplate.from_template(
            """다음 텍스트는 회사 규정 문서의 일부입니다.
이 문서는 아래 카테고리 중 어디에 해당하나요?

- 보안 규정
- 복리후생 규정
- 복무 규정
- 안전보건관리 규정
- 여비 규정
- 인사 규정
- 인사팀 업무 가이드
- 전산팀 업무 가이드
- 회계 규정
- 회계팀 업무 가이드

텍스트:
{text}

가장 관련 있는 카테고리 하나만 선택해서 정확히 출력하세요."""
        )
        
        # OpenAI LLM (카테고리 분류용)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        
    def create_collection(self):
        """컬렉션 생성"""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"컬렉션 '{self.collection_name}' 생성 완료")
            return True
        except Exception as e:
            logger.warning(f"컬렉션 생성 실패 (이미 존재할 수 있음): {e}")
            return False
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF에서 텍스트 추출"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"PDF 텍스트 추출 실패: {e}")
            return ""
    
    def strip_header(self, text: str) -> str:
        """문서 헤더 제거 (제1장, 제1조 등)"""
        match = re.search(r"제\s*\d+\s*[장|조]", text)
        return text[match.start():] if match else text
    
    def extract_category(self, filename: str, content: str) -> str:
        """파일명과 내용을 기반으로 카테고리 추출"""
        # 파일명 기반 카테고리 추출
        for cat, keywords in self.category_keywords.items():
            if any(kw in filename for kw in keywords):
                return cat
        
        # 내용 기반 카테고리 추출 (LLM 사용)
        try:
            response = self.llm.invoke(
                self.category_prompt.format(text=content[:1000])
            ).content.strip()
            
            if response in self.category_keywords.keys():
                return response
        except Exception as e:
            logger.warning(f"LLM 카테고리 분류 실패: {e}")
        
        return "기타"
    
    def extract_register_date(self, filename: str) -> str:
        """파일명에서 등록일 추출 (예: (240715))"""
        match = re.search(r"\((\d{6})\)", filename)
        return match.group(1) if match else "unknown"
    
    def add_document(self, pdf_path: str, document_name: str, batch_size: int = 100) -> bool:
        """문서를 벡터화하여 Qdrant에 추가 (배치 처리)"""
        try:
            # PDF 텍스트 추출
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                return False
            
            # 헤더 제거
            main_text = self.strip_header(text)
            
            # 카테고리 및 메타데이터 추출
            category = self.extract_category(document_name, main_text)
            register_date = self.extract_register_date(document_name)
            doc_title = document_name.replace(".pdf", "")
            
            # 텍스트 청킹
            chunks = self.text_splitter.split_text(main_text)
            
            # 배치 처리
            points = []
            for i, chunk in enumerate(chunks):
                try:
                    # 텍스트 임베딩
                    embedding = self.embedding_model.embed_query(chunk)
                    
                    # 포인트 생성
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "document_name": document_name,
                            "chunk_index": i,
                            "pdf_path": pdf_path,
                            "category": category,
                            "register_date": register_date,
                            "doc_title": doc_title
                        }
                    )
                    points.append(point)
                    
                    # 배치 크기에 도달하면 업로드
                    if len(points) >= batch_size:
                        self.client.upsert(
                            collection_name=self.collection_name,
                            points=points
                        )
                        points = []
                        
                except Exception as e:
                    logger.error(f"청크 {i} 처리 실패: {e}")
                    continue
            
            # 남은 포인트 처리
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            
            logger.info(f"문서 '{document_name}' 추가 완료 ({len(chunks)}개 청크, 카테고리: {category})")
            return True
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
            return False
    
    def search_similar(self, query: str, limit: int = 5, category: str = None) -> List[Dict[str, Any]]:
        """유사한 문서 검색 (카테고리 필터링 지원)"""
        try:
            # 쿼리 벡터화
            query_vector = self.embedding_model.embed_query(query)
            
            # 검색 조건 설정
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": query_vector,
                "limit": limit
            }
            
            # 카테고리 필터링
            if category:
                search_params["query_filter"] = {
                    "must": [
                        {
                            "key": "category",
                            "match": {"value": category}
                        }
                    ]
                }
            
            # 유사도 검색
            search_result = self.client.search(**search_params)
            
            # 결과 정리
            results = []
            for hit in search_result:
                results.append({
                    "score": hit.score,
                    "text": hit.payload["text"],
                    "document_name": hit.payload["document_name"],
                    "chunk_index": hit.payload["chunk_index"],
                    "pdf_path": hit.payload["pdf_path"],
                    "category": hit.payload.get("category", "기타"),
                    "register_date": hit.payload.get("register_date", "unknown"),
                    "doc_title": hit.payload.get("doc_title", "")
                })
            
            return results
            
        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.name,
                "vectors_count": info.vectors_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"컬렉션 정보 조회 실패: {e}")
            return {}
    
    def get_categories(self) -> List[str]:
        """사용 가능한 카테고리 목록 반환"""
        return list(self.category_keywords.keys())
    
    def delete_collection(self) -> bool:
        """컬렉션 삭제"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"컬렉션 '{self.collection_name}' 삭제 완료")
            return True
        except Exception as e:
            logger.error(f"컬렉션 삭제 실패: {e}")
            return False
