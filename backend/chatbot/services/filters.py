"""
도메인 분류 및 Qdrant 필터 생성
새로운 메타데이터 구조를 활용한 향상된 필터링
"""

from typing import List, Dict, Any, Optional
import re

# 도메인 분류 키워드 (새로운 메타데이터 구조에 맞게 업데이트)
DOMAIN_KEYWORDS = {
    '인사관리': [
        '인사', '급여', '채용', '복무', '교육', '훈련', '휴가', '연차', '출장', 
        '육아휴직', '성과평가', '승진', '이동', '퇴직', '연금', '복리후생'
    ],
    '재무관리': [
        '회계', '예산', '감사', '재정', '계약', '수수료', '자산', '장부', 
        '감사인', '계약사무', '비유동자산', '재무제표', '세무', '부채'
    ],
    '보안관리': [
        '보안', '정보보호', '정보보안', '개인정보', '민원', '신고', '보안정책',
        '접근통제', '암호화', '백업', '복구', '침해대응', '보안사고'
    ],
    '기술관리': [
        '기술', 'IT', '정보화', '전자서명', '시스템', '소프트웨어', '하드웨어',
        '네트워크', '데이터베이스', '클라우드', 'AI', '빅데이터', '블록체인'
    ],
    '행정관리': [
        '문서', '자료', '기록', '홍보', '문서관리', '자료관리', '기록물',
        '보존', '폐기', '이관', '공개', '비공개', '기밀'
    ],
    '경영관리': [
        '경영', '성과', '내부통제', '조직', '직제', '이해충돌', '윤리',
        '지속가능', 'ESG', '리스크', '품질', '혁신', '전략'
    ]
}

def guess_domains_from_keywords(keywords: List[str]) -> List[str]:
    """
    키워드를 기반으로 도메인을 추정
    
    Args:
        keywords: 추출된 키워드 리스트
    
    Returns:
        추정된 도메인 리스트 (신뢰도 순)
    """
    if not keywords:
        return []
    
    domain_scores = {}
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        
        for domain, domain_keywords in DOMAIN_KEYWORDS.items():
            for domain_keyword in domain_keywords:
                if domain_keyword in keyword_lower:
                    domain_scores[domain] = domain_scores.get(domain, 0) + 1
    
    # 점수 순으로 정렬
    sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 점수가 0보다 큰 도메인만 반환
    return [domain for domain, score in sorted_domains if score > 0]

def get_domain_keywords(domain: str) -> List[str]:
    """
    특정 도메인의 키워드 반환
    
    Args:
        domain: 도메인명
    
    Returns:
        해당 도메인의 키워드 리스트
    """
    return DOMAIN_KEYWORDS.get(domain, [])

def build_qdrant_filter(domain_list: List[str] = None, 
                       file_types: List[str] = None,
                       min_recency: int = None,
                       year_range: tuple = None) -> Optional[Dict[str, Any]]:
    """
    Qdrant 검색 필터 생성 (새로운 메타데이터 구조 활용)
    
    Args:
        domain_list: 도메인 리스트
        file_types: 문서 타입 리스트
        min_recency: 최소 최신성 점수 (1-3)
        year_range: 연도 범위 (start_year, end_year)
    
    Returns:
        Qdrant 필터 딕셔너리
    """
    filters = []
    
    # 도메인 필터
    if domain_list:
        domain_filter = {
            "key": "domain_primary",
            "match": {"any": domain_list}
        }
        filters.append(domain_filter)
    
    # 문서 타입 필터
    if file_types:
        type_filter = {
            "key": "document_type",
            "match": {"any": file_types}
        }
        filters.append(type_filter)
    
    # 최신성 점수 필터
    if min_recency and 1 <= min_recency <= 3:
        recency_filter = {
            "key": "recency_score",
            "range": {"gte": min_recency}
        }
        filters.append(recency_filter)
    
    # 연도 범위 필터
    if year_range and len(year_range) == 2:
        start_year, end_year = year_range
        if start_year and end_year and start_year <= end_year:
            year_filter = {
                "key": "year",
                "range": {"gte": start_year, "lte": end_year}
            }
            filters.append(year_filter)
    
    if filters:
        return {"must": filters}
    
    return None

def build_advanced_filter(query: str, 
                         estimated_domains: List[str] = None,
                         confidence_threshold: float = 0.7) -> Optional[Dict[str, Any]]:
    """
    고급 필터 생성 (질문 분석 기반)
    
    Args:
        query: 사용자 질문
        estimated_domains: 추정된 도메인
        confidence_threshold: 신뢰도 임계값
    
    Returns:
        고급 필터 딕셔너리
    """
    query_lower = query.lower()
    
    # 1. 도메인 기반 필터
    domain_filter = None
    if estimated_domains and len(estimated_domains) == 1:
        # 단일 도메인이 확실한 경우
        domain_filter = {
            "key": "domain_primary",
            "match": {"value": estimated_domains[0]}
        }
    
    # 2. 문서 타입 기반 필터
    type_filter = None
    doc_type_patterns = {
        '정관': ['정관', '기본법', '조직법'],
        '규정': ['규정', '운영규정', '관리규정'],
        '규칙': ['규칙', '세부규칙', '실행규칙'],
        '지침': ['지침', '업무지침', '운영지침']
    }
    
    for doc_type, keywords in doc_type_patterns.items():
        if any(keyword in query_lower for keyword in keywords):
            type_filter = {
                "key": "document_type",
                "match": {"value": doc_type}
            }
            break
    
    # 3. 최신성 기반 필터
    recency_filter = None
    recency_keywords = ['최신', '최근', '새로운', '업데이트', '변경', '수정']
    if any(keyword in query_lower for keyword in recency_keywords):
        recency_filter = {
            "key": "recency_score",
            "range": {"gte": 2}  # 최신성 점수 2 이상
        }
    
    # 4. 필터 조합
    filters = [f for f in [domain_filter, type_filter, recency_filter] if f is not None]
    
    if filters:
        return {"must": filters}
    
    return None

def get_filter_description(filter_dict: Dict[str, Any]) -> str:
    """
    필터에 대한 설명 생성
    
    Args:
        filter_dict: Qdrant 필터 딕셔너리
    
    Returns:
        필터 설명 문자열
    """
    if not filter_dict or 'must' not in filter_dict:
        return "필터 없음"
    
    descriptions = []
    
    for filter_item in filter_dict['must']:
        key = filter_item.get('key', '')
        match = filter_item.get('match', {})
        range_filter = filter_item.get('range', {})
        
        if match:
            if 'value' in match:
                descriptions.append(f"{key}: {match['value']}")
            elif 'any' in match:
                descriptions.append(f"{key}: {', '.join(match['any'])}")
        
        elif range_filter:
            if 'gte' in range_filter and 'lte' in range_filter:
                descriptions.append(f"{key}: {range_filter['gte']}~{range_filter['lte']}")
            elif 'gte' in range_filter:
                descriptions.append(f"{key}: {range_filter['gte']} 이상")
            elif 'lte' in range_filter:
                descriptions.append(f"{key}: {range_filter['lte']} 이하")
    
    return " | ".join(descriptions) if descriptions else "필터 없음"

def suggest_filters(query: str, keywords: List[str]) -> Dict[str, Any]:
    """
    질문과 키워드를 기반으로 필터 제안
    
    Args:
        query: 사용자 질문
        keywords: 추출된 키워드
    
    Returns:
        제안된 필터 정보
    """
    suggestions = {
        'domains': [],
        'file_types': [],
        'recency': None,
        'year_range': None,
        'confidence': 'low'
    }
    
    # 도메인 제안
    estimated_domains = guess_domains_from_keywords(keywords)
    if estimated_domains:
        suggestions['domains'] = estimated_domains
        suggestions['confidence'] = 'medium' if len(estimated_domains) == 1 else 'low'
    
    # 문서 타입 제안
    query_lower = query.lower()
    doc_type_keywords = {
        '정관': ['정관', '기본법', '조직법'],
        '규정': ['규정', '운영규정', '관리규정'],
        '규칙': ['규칙', '세부규칙', '실행규칙'],
        '지침': ['지침', '업무지침', '운영지침']
    }
    
    for doc_type, type_keywords in doc_type_keywords.items():
        if any(keyword in query_lower for keyword in type_keywords):
            suggestions['file_types'].append(doc_type)
    
    # 최신성 제안
    recency_keywords = ['최신', '최근', '새로운', '업데이트', '변경', '수정']
    if any(keyword in query_lower for keyword in recency_keywords):
        suggestions['recency'] = 2
    
    # 연도 범위 제안 (키워드에서 연도 추출)
    year_pattern = r'\b(20\d{2})\b'
    years = re.findall(year_pattern, query)
    if len(years) >= 2:
        years = sorted([int(y) for y in years])
        suggestions['year_range'] = (years[0], years[-1])
    elif len(years) == 1:
        year = int(years[0])
        suggestions['year_range'] = (year, year)
    
    return suggestions 