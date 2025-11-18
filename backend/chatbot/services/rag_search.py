"""
RAG 검색기
도메인 분류를 활용한 고급 검색을 담당합니다.
새로운 메타데이터 구조를 활용한 향상된 검색을 제공합니다.
"""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from django.conf import settings
from .constants import RAG_CONFIG, EXISTING_COLLECTION
from .filters import build_qdrant_filter, build_advanced_filter
import logging
import os

logger = logging.getLogger(__name__)

# 모듈 수준에서 즉시 모델 로딩 (강력한 캐싱)
print("🔥 SentenceTransformer 모델 모듈 로딩 시작...")
_GLOBAL_EMBEDDER = SentenceTransformer("nlpai-lab/KoE5")
print("🔥 SentenceTransformer 모델 모듈 로딩 완료!")

def get_global_embedder():
    """전역 임베딩 모델 반환 (이미 로딩됨)"""
    logger.info("캐싱된 SentenceTransformer 모델 사용")
    return _GLOBAL_EMBEDDER

class RagSearcher:
    """
    도메인 분류 기반 RAG 검색기
    새로운 메타데이터 구조를 활용한 고급 검색
    """
    
    def __init__(self, qdrant_host: str = None, qdrant_port: int = None, collection_name: str = None):
        """
        RAG 검색기 초기화
        
        Args:
            qdrant_host: Qdrant 호스트 (기본값: settings에서 가져옴)
            qdrant_port: Qdrant 포트 (기본값: settings에서 가져옴)
            collection_name: 컬렉션 이름 (기본값: 기존 컬렉션)
        """
        # Qdrant 클라이언트 초기화
        self.qdrant_host = qdrant_host or getattr(settings, 'QDRANT_HOST', 'qdrant')
        self.qdrant_port = qdrant_port or getattr(settings, 'QDRANT_PORT', 6333)
        self.collection_name = collection_name or EXISTING_COLLECTION
        
        self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        
        # 전역 캐싱된 임베딩 모델 사용
        self.embedder = get_global_embedder()
        
        # 검색 설정
        self.default_top_k = RAG_CONFIG.get('CHUNK_SIZE', 5)  # 5로 수정
    
    def search(self, query: str, flt: Optional[Dict[str, Any]] = None, top_k: int = None) -> List[Dict[str, Any]]:
        """
        질문에 대한 검색 수행 (새로운 메타데이터 구조 활용)
        
        Args:
            query: 검색 질문
            flt: Qdrant 필터
            top_k: 반환할 결과 수
        
        Returns:
            검색 결과 리스트
        """
        if top_k is None:
            top_k = self.default_top_k
        
        try:
            # 질문 임베딩
            query_vector = self.embedder.encode([query])[0].tolist()
            
            # Qdrant 검색
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=flt,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            
            # 새로운 메타데이터 구조에 맞게 결과 포맷팅
            formatted_results = []
            for result in search_results:
                payload = result.payload
                
                # 새로운 메타데이터 구조 활용
                formatted_result = {
                    'id': str(result.id),
                    'score': result.score,
                    'text': payload.get('text', ''),
                    'file_name': payload.get('doc_title', ''),
                    'pages': payload.get('page', ''),
                    'source': payload.get('source', ''),
                    
                    # 새로운 메타데이터 필드들
                    'document_level': payload.get('document_level', ''),
                    'document_type': payload.get('document_type', ''),
                    'domain_primary': payload.get('domain_primary', ''),
                    'domain_secondary': payload.get('domain_secondary', ''),
                    'year': payload.get('year', 0),
                    'month': payload.get('month', 0),
                    'day': payload.get('day', 0),
                    'recency_score': payload.get('recency_score', 1),
                    'total_pages': payload.get('total_pages', 0),
                    'chunk_index': payload.get('chunk_index', 0),
                    'total_chunks': payload.get('total_chunks', 0),
                    'chunk_char_len': payload.get('chunk_char_len', 0),
                    'register_date_iso': payload.get('register_date_iso', ''),
                    
                    # 기존 호환성 유지
                    'file_path': payload.get('file_path', ''),
                    'doc_id': payload.get('doc_id', ''),
                }
                
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            print(f"검색 오류: {e}")
            return []
    
    def search_by_domain(self, query: str, domain: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        특정 도메인으로 제한된 검색
        
        Args:
            query: 검색 질문
            domain: 도메인 (예: '인사관리', '재무관리')
            top_k: 반환할 결과 수
        
        Returns:
            도메인별 검색 결과
        """
        # 도메인 기반 필터 생성
        domain_filter = {
            "must": [
                {
                    "key": "domain_primary",
                    "match": {"value": domain}
                }
            ]
        }
        
        return self.search(query, flt=domain_filter, top_k=top_k)
    
    def search_by_file_type(self, query: str, file_type: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        특정 문서 타입으로 제한된 검색
        
        Args:
            query: 검색 질문
            file_type: 문서 타입 (예: '정관', '규정', '규칙', '지침')
            top_k: 반환할 결과 수
        
        Returns:
            문서 타입별 검색 결과
        """
        # 문서 타입 기반 필터 생성
        type_filter = {
            "must": [
                {
                    "key": "document_type",
                    "match": {"value": file_type}
                }
            ]
        }
        
        return self.search(query, flt=type_filter, top_k=top_k)
    
    def search_by_recency(self, query: str, min_recency: int = 1, top_k: int = None) -> List[Dict[str, Any]]:
        """
        최신성 점수 기반 검색
        
        Args:
            query: 검색 질문
            min_recency: 최소 최신성 점수 (1-3)
            top_k: 반환할 결과 수
        
        Returns:
            최신성 기반 검색 결과
        """
        # 최신성 점수 기반 필터 생성
        recency_filter = {
            "must": [
                {
                    "key": "recency_score",
                    "range": {"gte": min_recency}
                }
            ]
        }
        
        return self.search(query, flt=recency_filter, top_k=top_k)
    
    def search_forms(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        서식 전용 검색 (form_title, topics, synonyms 활용)
        
        Args:
            query: 검색 질문
            top_k: 반환할 결과 수
        
        Returns:
            서식 검색 결과
        """
        if top_k is None:
            top_k = self.default_top_k
        
        try:
            # 질문 임베딩
            query_vector = self.embedder.encode([query])[0].tolist()
            
            # 서식 전용 필터 (doc_type이 "form"인 것만)
            form_filter = {
                "must": [
                    {
                        "key": "doc_type",
                        "match": {"value": "form"}
                    }
                ]
            }
            
            # Qdrant 검색
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=form_filter,
                limit=top_k * 2,  # 더 많이 검색해서 재순위화
                with_payload=True,
                with_vectors=False
            )
            
            # 서식 메타데이터를 활용한 재순위화
            reranked_results = self._rerank_forms(search_results, query)
            
            # 상위 결과만 반환
            return reranked_results[:top_k]
            
        except Exception as e:
            print(f"서식 검색 오류: {e}")
            return []
    
    def _rerank_forms(self, results: List, query: str) -> List[Dict[str, Any]]:
        """
        서식 검색 결과 재순위화 (form_title, topics, synonyms 활용)
        
        Args:
            results: 원본 검색 결과
            query: 검색 질문
        
        Returns:
            재순위화된 서식 결과
        """
        query_lower = query.lower()
        reranked = []
        
        for result in results:
            payload = result.payload
            
            # 서식 관련 메타데이터 추출
            form_title = payload.get('form_title', '')
            topics = payload.get('topics', [])
            synonyms = payload.get('synonyms', [])
            
            # 관련성 점수 계산
            relevance_score = 0
            
            # 1. 서식 제목 직접 매칭 (높은 가중치)
            if form_title and any(keyword in form_title.lower() for keyword in query_lower.split()):
                relevance_score += 5
            
            # 2. 토픽 매칭
            for topic in topics:
                if topic and any(keyword in topic.lower() for keyword in query_lower.split()):
                    relevance_score += 3
            
            # 3. 동의어 매칭
            for synonym in synonyms:
                if synonym and any(keyword in synonym.lower() for keyword in query_lower.split()):
                    relevance_score += 2
            
            # 4. 벡터 유사도 점수
            relevance_score += result.score * 2
            
            # 새로운 메타데이터 구조에 맞게 결과 포맷팅
            formatted_result = {
                'id': str(result.id),
                'score': result.score,
                'relevance_score': relevance_score,
                'text': payload.get('text', ''),
                'file_name': payload.get('doc_title', ''),
                'pages': payload.get('page', ''),
                'source': payload.get('source', ''),
                
                # 서식 관련 메타데이터
                'form_title': form_title,
                'form_page': payload.get('form_page', ''),
                'form_file_uri': payload.get('form_file_uri', ''),
                'topics': topics,
                'synonyms': synonyms,
                'anchor_refs': payload.get('anchor_refs', []),
                
                # 기존 메타데이터
                'document_level': payload.get('document_level', ''),
                'document_type': payload.get('document_type', ''),
                'domain_primary': payload.get('domain_primary', ''),
                'domain_secondary': payload.get('domain_secondary', ''),
                'year': payload.get('year', 0),
                'recency_score': payload.get('recency_score', 1),
                'file_path': payload.get('file_path', ''),
                'doc_id': payload.get('doc_id', ''),
            }
            
            reranked.append(formatted_result)
        
        # 관련성 점수로 재정렬
        reranked.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return reranked

    def hybrid_search(self, query: str, domain_list: List[str] = None, 
                     file_types: List[str] = None, min_recency: int = None,
                     top_k: int = None) -> List[Dict[str, Any]]:
        """
        하이브리드 검색 (벡터 + 메타데이터 필터링)
        
        Args:
            query: 검색 질문
            domain_list: 도메인 리스트
            file_types: 문서 타입 리스트
            min_recency: 최소 최신성 점수
            top_k: 반환할 결과 수
        
        Returns:
            하이브리드 검색 결과
        """
        # 복합 필터 생성
        filters = []
        
        if domain_list:
            domain_filter = {
                "key": "domain_primary",
                "match": {"any": domain_list}
            }
            filters.append(domain_filter)
        
        if file_types:
            type_filter = {
                "key": "document_type",
                "match": {"any": file_types}
            }
            filters.append(type_filter)
        
        if min_recency:
            recency_filter = {
                "key": "recency_score",
                "range": {"gte": min_recency}
            }
            filters.append(recency_filter)
        
        # 필터 적용
        if filters:
            query_filter = {"must": filters}
        else:
            query_filter = None
        
        # 검색 실행
        results = self.search(query, flt=query_filter, top_k=top_k)
        
        # 결과 재순위화 (도메인 일치도 + 최신성 점수 고려)
        if results:
            results = self._rerank_results(results, query, domain_list)
        
        return results
    
    def _rerank_results(self, results: List[Dict[str, Any]], query: str, 
                       domain_list: List[str] = None) -> List[Dict[str, Any]]:
        """
        검색 결과 재순위화 (도메인 일치도 + 최신성 점수 고려)
        
        Args:
            results: 원본 검색 결과
            query: 검색 질문
            domain_list: 도메인 리스트
        
        Returns:
            재순위화된 결과
        """
        for result in results:
            # 도메인 일치도 점수 계산
            domain_score = 0
            if domain_list and result.get('domain_primary') in domain_list:
                domain_score = 2  # 도메인 일치 시 높은 점수
            
            # 최신성 점수 활용
            recency_score = result.get('recency_score', 1)
            
            # 최종 점수 계산 (벡터 유사도 + 도메인 일치도 + 최신성)
            final_score = result['score'] + domain_score + (recency_score * 0.1)
            result['final_score'] = final_score
        
        # 최종 점수로 재정렬
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        return results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        컬렉션 정보 조회
        
        Returns:
            컬렉션 메타데이터
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': info.name,
                'points_count': getattr(info, 'points_count', 0),
                'vectors_count': getattr(info, 'vectors_count', 0),
                'status': getattr(info, 'status', 'unknown')
            }
        except Exception as e:
            print(f"컬렉션 정보 조회 오류: {e}")
            return {}
    
    def health_check(self) -> bool:
        """
        Qdrant 연결 상태 확인
        
        Returns:
            연결 상태 (True/False)
        """
        try:
            collections = self.client.get_collections()
            return True
        except Exception as e:
            print(f"Qdrant 연결 오류: {e}")
            return False 