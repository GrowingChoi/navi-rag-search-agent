# NAVI RAG Search Agent 🧭  
한국인터넷진흥원(KISA) 규정 문서 기반 사내 업무 지원 RAG 챗봇

> 대규모 규정·지침·지침서 PDF를 벡터화하고,  
> 사내 직원의 질의를 RAG 파이프라인으로 처리해 **실무에 바로 쓸 수 있는 답변**을 제공하는 AI 검색 에이전트입니다.

---

## 1. 프로젝트 개요

한국인터넷진흥원(KISA)의 정관, 규정, 규칙, 지침, 서식류 등 수백 개의 PDF 문서를 대상으로

- “어떤 규정의 몇 조에 이런 내용이 있나요?”
- “출장비와 관련된 양식 서류 뭐 써야 하죠?”
- “안전보건관리규정에서 작업중지 관련 조항 찾아줘”

와 같은 실무성 높은 질문을 하면,

> **LLM + RAG 파이프라인 + Qdrant 벡터 DB**를 이용해  
> 관련 조항과 근거 문서를 함께 반환하는 검색 챗봇입니다.

이 레포는 기존 팀 프로젝트(SKN13-FINAL-6Team) 중  
**문서 검색 / RAG 에이전트 기능을 중심으로 재구성한 버전**입니다.

---

## 2. 주요 기능

### 🔎 규정 문서 RAG 검색

- `documents_kisa_pdf` 하위의 정관/규정/규칙/지침 PDF 전체를 벡터화
- 파일명 규칙을 이용해 **문서 유형·도메인 메타데이터 자동 부여**
  - 예: `2_05_인사규정(240221).pdf`
    - `2` → 규정
    - `05` → 인사 도메인
- 사용자의 자연어 질문을 벡터 검색 + 재랭킹 후, LLM으로 최종 답변 생성
- **출처 문서/페이지 정보**를 함께 반환해 신뢰성 확보

### 🧾 서식/양식(Form) 특화 검색

- `documents_kisa_pdf/forms_extracted_v6` 폴더의 각종 신청서/확인서/양식 문서를 별도 인덱싱
- “○○신청서 양식”, “○○확인서 서식” 등 **양식 관련 질의**는 일반 규정이 아니라  
  **form 문서 컬렉션만 타겟팅해서 검색**
- `backend/analyze_form_patterns.py`로 form 문서 패턴 분석, form 전용 검색 로직에 활용

### 🧠 RAG 파이프라인 구조

- `backend/chatbot/services/rag_indexer.py`
  - PDF → 텍스트 추출
  - 청크 분할(길이/오버랩)
  - 메타데이터(문서 유형, 도메인, 파일명, 페이지 등) 부착
  - Qdrant 컬렉션에 업로드

- `backend/chatbot/services/rag_search.py`
  - 사용자의 질의(q)를 벡터화
  - Qdrant에서 관련 문서 Top-k 검색
  - form 관련 질의 여부에 따라 **일반 규정 / form 문서 컬렉션 분기 처리**

- `backend/chatbot/services/pipeline.py`
  - 검색된 문서들을 컨텍스트로 묶어 LLM에 전달
  - `config/system_prompt.md`, `config/user_prompt.md`에 분리된 프롬프트를 로드
  - 최종 Answer + 출처 정보 구조화

- `backend/chatbot/services/answerer.py`
  - LLM 호출 래핑
  - 답변 포맷팅, 마크다운 변환, 강조/목록 처리 등 후처리 담당

- `backend/chatbot/services/constants.py`, `filters.py`, `keyword_extractor.py`
  - 컬렉션 이름, 메타데이터 키, 질의 패턴 기반 필터링/라우팅 로직 정의

- `backend/test_enhanced_rag.py`
  - RAG 검색 품질을 스크립트로 테스트/검증

---

## 3. 기술 스택

### Backend

- Python 3.x  
- Django, Django REST Framework 기반 앱 구조(`chatbot`, `qdrant` 등)  
- Qdrant (Vector DB 클라이언트 래퍼)  

### AI / RAG

- OpenAI GPT 계열 (예: GPT-4o-mini)  
- Embedding 모델: `text-embedding-3-large`  
- 커스텀 RAG 파이프라인
  - 문서 파서, 인덱서, 서치 엔진, 파이프라인 직접 구현

### Infra / 기타

- (원본 프로젝트 기준) Docker, Docker Compose, Nginx  
- 프론트엔드는 별도 레포/배포(예: Vercel)에서 React + Vite + Tailwind로 구현 가능

---

## 4. 폴더 구조 (RAG 관련 중심)

```bash
NAVI-RAG-SEARCH-AGENT/
├── backend/
│   ├── chatbot/
│   │   ├── services/
│   │   │   ├── answerer.py
│   │   │   ├── api.py
│   │   │   ├── constants.py
│   │   │   ├── filters.py
│   │   │   ├── keyword_extractor.py
│   │   │   ├── pipeline.py
│   │   │   ├── rag_indexer.py
│   │   │   ├── rag_search.py
│   │   │   ├── rag_service.py
│   │   │   └── __init__.py
│   │   │
│   │   ├── models.py
│   │   ├── serializers.py
│   │   ├── urls.py
│   │   ├── views.py
│   │   └── __init__.py
│   │
│   ├── qdrant/
│   │   ├── __init__.py
│   │   ├── admin.py
│   │   ├── apps.py
│   │   ├── models.py
│   │   ├── services.py
│   │   ├── tests.py
│   │   ├── urls.py
│   │   └── views.py
│   │
│   ├── analyze_form_patterns.py
│   ├── embed_documents.py
│   └── test_enhanced_rag.py
│
├── config/
│   ├── system_prompt.md
│   └── user_prompt.md
│
├── documents_kisa_pdf/
│   ├── forms_extracted_v6/
│   ├── 1_01_한국인터넷진흥원 정관(240715).pdf
│   ├── 2_05_인사규정(240221).pdf
│   ├── 3_05_인사관리규칙(230706).pdf
│   └── 4_03_교육훈련지침(240201).pdf
│
└── README.md
