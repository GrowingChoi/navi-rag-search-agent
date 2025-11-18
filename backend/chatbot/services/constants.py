"""
RAG 시스템 상수 정의
새로운 메타데이터 구조에 맞게 업데이트
"""

# 도메인 분류 체계 (새로운 메타데이터 구조 기반)
DOMAIN_CLASSIFICATION = {
    '인사관리': {
        'keywords': ['인사', '채용', '급여', '휴가', '육아휴직', '연차', '출장', '복무', '교육', '훈련'],
        'description': '인사 정책, 급여, 복무, 교육 등 인적자원 관리',
        'subdomains': ['급여관리', '채용관리', '복무관리', '교육훈련', '인사정책']
    },
    '재무관리': {
        'keywords': ['회계', '예산', '감사', '재정', '계약', '수수료', '자산'],
        'description': '회계, 예산, 감사, 계약, 자산 등 재무 관련',
        'subdomains': ['회계관리', '감사관리', '계약관리', '자산관리', '재무정책']
    },
    '보안관리': {
        'keywords': ['보안', '정보보호', '개인정보', '민원', '신고'],
        'description': '보안 정책, 정보보호, 개인정보 관리',
        'subdomains': ['정보보호', '개인정보보호', '민원신고', '보안정책']
    },
    '기술관리': {
        'keywords': ['기술', 'IT', '정보화', '전자서명', '시스템'],
        'description': '기술 정책, IT 시스템, 정보화',
        'subdomains': ['정보화관리', '전자서명관리', '기술정책']
    },
    '행정관리': {
        'keywords': ['문서', '자료', '기록', '홍보'],
        'description': '문서 관리, 자료 관리, 홍보 업무',
        'subdomains': ['문서관리', '자료관리', '기록관리', '행정정책']
    },
    '경영관리': {
        'keywords': ['경영', '성과', '내부통제', '이해충돌', '조직', '직제'],
        'description': '경영 정책, 성과 관리, 조직 관리',
        'subdomains': ['성과관리', '내부통제', '조직관리', '경영정책']
    }
}

# 문서 타입 패턴 (새로운 메타데이터 구조 기반)
DOCUMENT_TYPE_PATTERNS = {
    '1_': {
        'type': '정관',
        'description': '기본법령',
        'level': 1,
        'priority': 'highest'
    },
    '2_': {
        'type': '규정',
        'description': '운영규정, 관리규정',
        'level': 2,
        'priority': 'high'
    },
    '3_': {
        'type': '규칙',
        'description': '세부규칙, 실행규칙',
        'level': 3,
        'priority': 'medium'
    },
    '4_': {
        'type': '지침',
        'description': '업무지침, 운영지침',
        'level': 4,
        'priority': 'low'
    }
}

# 도메인 힌트 (키워드 → 도메인 매핑)
DOMAIN_HINTS = {
    '급여': '인사관리',
    '채용': '인사관리',
    '복무': '인사관리',
    '교육': '인사관리',
    '회계': '재무관리',
    '예산': '재무관리',
    '감사': '재무관리',
    '계약': '재무관리',
    '보안': '보안관리',
    '정보보호': '보안관리',
    '개인정보': '보안관리',
    '기술': '기술관리',
    'IT': '기술관리',
    '정보화': '기술관리',
    '문서': '행정관리',
    '자료': '행정관리',
    '경영': '경영관리',
    '성과': '경영관리',
    '조직': '경영관리'
}

# 최신성 점수 임계값
RECENCY_THRESHOLDS = {
    'current': 3,      # 현재 연도
    'recent': 2,       # 최근 2년
    'legacy': 1        # 3년 이상
}

# RAG 시스템 설정
RAG_CONFIG = {
    'CHUNK_SIZE': 8,           # 기본 검색 결과 수
    'MAX_CONTEXT_LENGTH': 4000, # 최대 컨텍스트 길이
    'MIN_SCORE_THRESHOLD': 0.7, # 최소 유사도 점수
    'RERANK_ENABLED': True,     # 재순위화 활성화
    'DOMAIN_WEIGHT': 0.3,       # 도메인 가중치
    'RECENCY_WEIGHT': 0.1,      # 최신성 가중치
    'VECTOR_WEIGHT': 0.6        # 벡터 유사도 가중치
}

# 기존 컬렉션 이름 (호환성 유지)
EXISTING_COLLECTION = "regulations_final"

# 검색 전략 설정
SEARCH_STRATEGIES = {
    'domain_specific': {
        'confidence': 'high',
        'filter_strength': 'strong',
        'fallback': 'hybrid'
    },
    'file_type_specific': {
        'confidence': 'medium',
        'filter_strength': 'medium',
        'fallback': 'hybrid'
    },
    'recency_aware': {
        'confidence': 'medium',
        'filter_strength': 'weak',
        'fallback': 'hybrid'
    },
    'hybrid': {
        'confidence': 'low',
        'filter_strength': 'weak',
        'fallback': 'vector_only'
    }
}

# 메타데이터 필드 매핑
METADATA_FIELDS = {
    'core': ['doc_id', 'doc_title', 'source', 'text', 'page'],
    'classification': ['document_level', 'document_type', 'domain_primary', 'domain_secondary'],
    'temporal': ['year', 'month', 'day', 'recency_score', 'register_date_iso'],
    'structural': ['total_pages', 'chunk_index', 'total_chunks', 'chunk_char_len'],
    'technical': ['embed_backend', 'embed_model', 'embed_dim', 'file_size']
}

# 필터 우선순위
FILTER_PRIORITY = {
    'domain_primary': 1,
    'document_type': 2,
    'recency_score': 3,
    'year': 4,
    'page': 5
} 