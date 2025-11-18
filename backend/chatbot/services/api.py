"""
RAG 시스템 Django API 뷰
기존 API와 충돌하지 않도록 별도 네임스페이스로 구현합니다.
"""

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from .pipeline import answer_query, quick_search, get_domain_suggestions, health_check
from .constants import DOMAIN_CLASSIFICATION

@api_view(['POST'])
@permission_classes([AllowAny])
def rag_ask(request):
    """
    RAG 질의 응답 API
    
    요청:
    {
        "query": "질문 내용",
        "domain": "도메인명" (선택사항)
    }
    
    응답:
    {
        "success": true,
        "answer": "답변 내용",
        "used_domains": ["도메인1", "도메인2"],
        "top_docs": [
            {
                "file_name": "파일명",
                "pages": "페이지",
                "domain": "도메인",
                "score": 0.95,
                "text_preview": "텍스트 미리보기..."
            }
        ],
        "keywords": ["키워드1", "키워드2"],
        "quality": {
            "score": 8,
            "issues": [],
            "quality_level": "우수"
        }
    }
    """
    try:
        query = request.data.get('query', '').strip()
        domain = request.data.get('domain', '').strip()
        
        # 입력 검증
        if not query:
            return Response({
                'success': False,
                'error': '질문을 입력해주세요.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # 도메인 유효성 검사
        if domain and domain not in DOMAIN_CLASSIFICATION:
            return Response({
                'success': False,
                'error': f'유효하지 않은 도메인입니다. 사용 가능한 도메인: {list(DOMAIN_CLASSIFICATION.keys())}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # RAG 파이프라인 실행
        result = answer_query(query, explicit_domain=domain if domain else None)
        
        return Response(result, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'success': False,
            'error': f'서버 오류가 발생했습니다: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([AllowAny])
def rag_search(request):
    """
    빠른 검색 API (답변 생성 없음)
    
    쿼리 파라미터:
    - query: 검색 질문
    - domain: 도메인 (선택사항)
    - top_k: 결과 수 (기본값: 5)
    """
    try:
        query = request.GET.get('query', '').strip()
        domain = request.GET.get('domain', '').strip()
        top_k = int(request.GET.get('top_k', 5))
        
        if not query:
            return Response({
                'success': False,
                'error': '검색어를 입력해주세요.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # 빠른 검색 실행
        results = quick_search(query, domain if domain else None, top_k)
        
        return Response({
            'success': True,
            'query': query,
            'domain': domain,
            'results': results,
            'count': len(results)
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'success': False,
            'error': f'검색 중 오류가 발생했습니다: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([AllowAny])
def rag_domains(request):
    """
    도메인 정보 및 제안 API
    
    쿼리 파라미터:
    - query: 질문 (도메인 제안을 위한)
    """
    try:
        query = request.GET.get('query', '').strip()
        
        # 전체 도메인 정보
        domains_info = {}
        for domain, info in DOMAIN_CLASSIFICATION.items():
            domains_info[domain] = {
                'description': info['description'],
                'keywords': info['keywords']
            }
        
        # 질문이 있으면 도메인 제안
        suggested_domains = []
        if query:
            suggested_domains = get_domain_suggestions(query)
        
        return Response({
            'success': True,
            'domains': domains_info,
            'suggested_domains': suggested_domains,
            'query': query
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'success': False,
            'error': f'도메인 정보 조회 중 오류가 발생했습니다: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([AllowAny])
def rag_health(request):
    """
    RAG 시스템 상태 확인 API
    """
    try:
        health_info = health_check()
        return Response(health_info, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'status': 'error',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([AllowAny])
def rag_config(request):
    """
    RAG 시스템 설정 정보 API
    """
    try:
        from .constants import RAG_CONFIG, EXISTING_COLLECTION
        
        config_info = {
            'collection_name': EXISTING_COLLECTION,
            'chunk_size': RAG_CONFIG['CHUNK_SIZE'],
            'chunk_overlap': RAG_CONFIG['CHUNK_OVERLAP'],
            'min_doc_chars': RAG_CONFIG['MIN_DOC_CHARS'],
            'min_page_chars': RAG_CONFIG['MIN_PAGE_CHARS'],
            'min_chunk_chars': RAG_CONFIG['MIN_CHUNK_CHARS'],
            'batch_size': RAG_CONFIG['BATCH_SIZE'],
            'version': RAG_CONFIG['VERSION'],
            'available_domains': list(DOMAIN_CLASSIFICATION.keys())
        }
        
        return Response({
            'success': True,
            'config': config_info
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'success': False,
            'error': f'설정 정보 조회 중 오류가 발생했습니다: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR) 