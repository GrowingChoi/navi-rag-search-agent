"""
RAG 시스템 통합 파이프라인
전체 RAG 워크플로우를 관리합니다.
새로운 메타데이터 구조를 활용한 향상된 검색과 답변 생성
"""

from typing import List, Dict, Any, Optional
from django.conf import settings
from .keyword_extractor import extract_keywords
from .filters import guess_domains_from_keywords
from .rag_search import RagSearcher
from .answerer import make_answer, format_context_for_display, validate_answer_quality
import datetime
import re
import os
import sys
import time
import logging
import openai
import hashlib

# 로깅 설정
logger = logging.getLogger(__name__)

# 프롬프트 로더 직접 구현
def load_prompt(path: str, *, default: str = "") -> str:
    """프롬프트 파일을 로드하는 함수"""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"WARNING: Failed to load prompt from {path}: {e}")
        return default

# 전역 프롬프트 변수
_SYSTEM_PROMPT = None
_USER_PROMPT = None

def _init_prompts():
    """프롬프트 초기화 함수"""
    global _SYSTEM_PROMPT, _USER_PROMPT
    
    if _SYSTEM_PROMPT is None:
        try:
            system_prompt_path = '/app/config/system_prompt.md'
            _SYSTEM_PROMPT = load_prompt(system_prompt_path,
                                         default="당신은 업무 가이드를 제공하는 전문가입니다.")
        except FileNotFoundError:
            _SYSTEM_PROMPT = "당신은 업무 가이드를 제공하는 전문가입니다."
            print("WARNING: system_prompt.md not found, using default prompt")
    
    if _USER_PROMPT is None:
        try:
            user_prompt_path = '/app/config/user_prompt.md'
            _USER_PROMPT = load_prompt(user_prompt_path,
                                       default="위 문서들을 바탕으로 질문에 대한 정확한 답변을 제공해주세요.")
        except FileNotFoundError:
            _USER_PROMPT = "위 문서들을 바탕으로 질문에 대한 정확한 답변을 제공해주세요."
            print("WARNING: user_prompt.md not found, using default prompt")
    
    return _SYSTEM_PROMPT, _USER_PROMPT

def analyze_user_input(query: str, openai_api_key: str = None) -> Dict[str, Any]:
    """
    사용자 입력을 종합적으로 분석하여 모든 정보를 한 번에 추출
    
    Args:
        query: 사용자 질문
        openai_api_key: OpenAI API 키
    
    Returns:
        {
            'is_simple_greeting': bool,
            'is_department_intro': bool,
            'department': str or None,
            'user_info': {'department': str, 'position': str, 'name': str} or None
        }
    """
    try:
        # OpenAI 클라이언트 설정
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            return {
                'is_simple_greeting': True,
                'is_department_intro': False,
                'department': None,
                'user_info': None
            }
        
        client = openai.OpenAI(api_key=api_key)
        
        # 종합 분석 프롬프트
        system_prompt = """당신은 사용자의 입력을 종합적으로 분석하는 전문가입니다. 다음을 모두 분석해주세요:

1. 간단한 인사말/대화인지 판단:
- 간단한 인사말: "안녕하세요", "안녕", "Hi", "좋은 아침", "감사합니다" 등
- 자기소개 포함 인사말: "안녕, 나는 김철수야", "개발팀에서 일해요, 안녕" 등

2. 부서/팀 소개인지 판단:
- "나는 개발팀이야", "개발팀에서 일해요", "개발팀 김○○입니다"
- "인사팀에서 일해요", "IT팀입니다", "개발부서에서 근무합니다" 등
- "개발팀 업무 알려줘", "개발팀에서 뭘 해야해?", "개발팀 업무가 궁금해" 등
- "우리부서에 도움이 되려면", "우리팀에서 필요한 것", "우리 조직에 필요한 스킬" 등
- 부서명 + 업무/일/해야할 것 등의 조합도 부서 소개로 인식

3. 사용자 정보 추출:
- 부서/팀: 사용자가 언급한 모든 부서/팀명 (예: 개발팀, 인사팀, 회계팀, 전산팀, IT팀, 기획팀, 마케팅팀, 개발부서, 기술팀, 소프트웨어팀, 연구팀, 운영팀, 보안팀, 품질팀, 영업팀, 고객지원팀 등)
- 직급: 사용자가 언급한 모든 직급 (예: 사원, 대리, 과장, 차장, 부장, 이사, 상무, 전무, 사장, 팀장, 부서장, 본부장, 대표이사 등)
- 이름: 사용자가 언급한 성명 (한글, 영문 모두 가능)

중요: 
- "개발팀 업무 알려줘" 같은 질문에서 "개발팀"을 부서로 인식하고, is_department_intro를 true로 설정하세요.
- "우리부서", "우리팀", "우리 조직" 같은 표현도 부서 소개로 인식하세요.
- 기존 대화에서 사용자가 언급한 부서 정보를 활용하세요.

답변 형식 (JSON):
{
  "is_simple_greeting": true/false,
  "is_department_intro": true/false,
  "department": "부서명 또는 null",
  "user_info": {
    "department": "부서명 또는 null",
    "position": "직급 또는 null",
    "name": "이름 또는 null"
  }
}

예시:
- "안녕하세요" → {"is_simple_greeting": true, "is_department_intro": false, "department": null, "user_info": null}
- "안녕, 나는 김철수야" → {"is_simple_greeting": true, "is_department_intro": false, "department": null, "user_info": {"department": null, "position": null, "name": "김철수"}}
- "개발팀에서 일해요" → {"is_simple_greeting": true, "is_department_intro": true, "department": "개발팀", "user_info": {"department": "개발팀", "position": null, "name": null}}
- "안녕, 나는 김철수, 개발팀이야" → {"is_simple_greeting": true, "is_department_intro": true, "department": "개발팀", "user_info": {"department": "개발팀", "position": null, "name": "김철수"}}"""

        user_prompt = f"다음 입력을 종합적으로 분석해주세요: '{query}'"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )
        
        result = response.choices[0].message.content.strip()
        
        # JSON 파싱 시도
        import json
        try:
            analysis = json.loads(result)
            # 빈 문자열을 None으로 변환
            if analysis.get('user_info'):
                for key in analysis['user_info']:
                    if analysis['user_info'][key] == "" or analysis['user_info'][key] == "null":
                        analysis['user_info'][key] = None
            return analysis
        except json.JSONDecodeError:
            return {
                'is_simple_greeting': True,
                'is_department_intro': False,
                'department': None,
                'user_info': None
            }
        
    except Exception as e:
        logger.error(f"사용자 입력 분석 중 오류: {e}")
        return {
            'is_simple_greeting': True,
            'is_department_intro': False,
            'department': None,
            'user_info': None
        }

def is_simple_greeting(query: str, openai_api_key: str = None) -> bool:
    """
    통합된 사용자 입력 분석 함수를 사용하여 간단한 인사말인지 판단
    
    Args:
        query: 사용자 질문
        openai_api_key: OpenAI API 키
    
    Returns:
        True if simple greeting, False if complex question
    """
    analysis = analyze_user_input(query, openai_api_key)
    return analysis.get('is_simple_greeting', True)

def update_user_context(conversation_history: List[Dict], user_info: Dict[str, str]) -> List[Dict]:
    """
    대화 히스토리에 사용자 정보를 업데이트
    
    Args:
        conversation_history: 기존 대화 히스토리
        user_info: 추출된 사용자 정보
    
    Returns:
        업데이트된 대화 히스토리
    """
    if not conversation_history:
        conversation_history = []
    
    print(f"DEBUG: 사용자 정보 저장 시작 - user_info: {user_info}")
    print(f"DEBUG: 기존 대화 히스토리 길이: {len(conversation_history)}")
    
    # 기존 사용자 정보 찾기
    user_context = None
    for i, msg in enumerate(conversation_history):
        if isinstance(msg, dict) and msg.get("role") == "system" and "user_context" in msg.get("content", ""):
            user_context = msg
            print(f"DEBUG: 기존 사용자 정보 발견 (인덱스 {i}): {msg.get('content', '')}")
            break
    
    # 새로운 사용자 정보 생성
    import json
    context_content = f"user_context: {json.dumps(user_info, ensure_ascii=False)}"
    print(f"DEBUG: 생성된 컨텍스트 내용: {context_content}")
    
    if user_context:
        # 기존 사용자 정보 업데이트
        user_context["content"] = context_content
        print(f"DEBUG: 기존 사용자 정보 업데이트 완료")
    else:
        # 새로운 사용자 정보 추가
        conversation_history.insert(0, {
            "role": "system",
            "content": context_content
        })
        print(f"DEBUG: 새로운 사용자 정보 추가 완료")
    
    print(f"DEBUG: 업데이트된 대화 히스토리 길이: {len(conversation_history)}")
    return conversation_history

def prioritize_results_by_department(search_results: List[Dict], user_department: str, openai_api_key: str = None) -> List[Dict]:
    """
    사용자 부서에 맞게 검색 결과 우선순위 조정 (LLM 기반 동적 처리)
    
    Args:
        search_results: 검색 결과 리스트
        user_department: 사용자 부서
        openai_api_key: OpenAI API 키
    
    Returns:
        우선순위가 조정된 검색 결과 리스트
    """
    if not user_department or not search_results:
        return search_results
    
    try:
        # LLM을 사용하여 부서별 관련 카테고리 동적 추출
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if api_key:
            client = openai.OpenAI(api_key=api_key)
            
            # 검색 결과에서 카테고리 정보 추출
            categories = list(set([result.get('category', '') for result in search_results if result.get('category')]))
            
            system_prompt = f"""당신은 한국인터넷진흥원(KISA)의 업무 가이드 전문가입니다.
사용자가 "{user_department}"에서 근무할 때, 다음 카테고리 중에서 가장 관련성이 높은 카테고리들을 우선순위 순으로 선택해주세요.

사용 가능한 카테고리:
{', '.join(categories)}

답변 형식: 관련성이 높은 순서대로 카테고리명을 쉼표로 구분하여 나열하세요.
예시: "인사 규정, 복리후생 규정, 인사팀 업무 가이드"

{user_department}와 가장 관련성이 높은 상위 3-5개 카테고리만 선택하세요."""

            user_prompt = f"{user_department}에서 근무하는 사용자에게 가장 관련성이 높은 카테고리를 선택해주세요."
            
            response_result = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            priority_categories_text = response_result.choices[0].message.content.strip()
            priority_categories = [cat.strip() for cat in priority_categories_text.split(',')]
            
        else:
            # API 키가 없는 경우 기본 우선순위 사용
            priority_categories = []
    
    except Exception as e:
        logger.error(f"부서별 우선순위 추출 실패: {e}")
        priority_categories = []
    
    # 우선순위별로 결과 분류
    high_priority = []
    medium_priority = []
    low_priority = []
    
    for result in search_results:
        category = result.get('category', '')
        is_high_priority = any(priority in category for priority in priority_categories)
        
        if is_high_priority:
            high_priority.append(result)
        elif category:
            medium_priority.append(result)
        else:
            low_priority.append(result)
    
    # 우선순위 순으로 재정렬
    prioritized_results = high_priority + medium_priority + low_priority
    
    logger.info(f"부서별 우선순위 조정: {user_department} - 고우선순위: {len(high_priority)}, 중우선순위: {len(medium_priority)}, 저우선순위: {len(low_priority)}")
    
    return prioritized_results

def analyze_question_level(query: str, openai_api_key: str = None) -> Dict[str, Any]:
    """
    질문의 수준과 예상 후속 질문을 분석
    
    Args:
        query: 사용자 질문
        openai_api_key: OpenAI API 키
    
    Returns:
        {'level': '기초/중급/고급', 'follow_up_questions': ['질문1', '질문2', ...]}
    """
    try:
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            return {'level': '중급', 'follow_up_questions': []}
        
        client = openai.OpenAI(api_key=api_key)
        
        system_prompt = """당신은 질문 분석 전문가입니다. 사용자의 질문을 분석하여 다음을 판단하세요:

1. 질문 수준 분류:
- 기초: 기본적인 개념이나 절차에 대한 질문 (예: "휴가 신청이 뭐야?", "급여는 언제 받나요?")
- 중급: 구체적인 업무 절차나 정책에 대한 질문 (예: "휴가 신청 절차는?", "연차 사용 규정은?")
- 고급: 복잡한 업무나 정책 해석에 대한 질문 (예: "특별휴가와 연차의 차이점은?", "급여 계산 방식은?")

2. 예상 후속 질문 생성:
질문 수준에 따라 사용자가 다음에 궁금해할 수 있는 내용을 유도형 질문으로 2-3개 생성하세요.

답변 형식 (JSON):
{
  "level": "기초/중급/고급",
  "follow_up_questions": ["혹시 ~에 대해 궁금하신가요?", "~에 대한 정보도 필요하실까요?", "~에 대해서도 알고 싶으시다면 말씀해 주세요."]
}"""

        user_prompt = f"다음 질문을 분석해주세요: '{query}'"
        
        response_result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        result = response_result.choices[0].message.content.strip()
        
        # JSON 파싱 시도
        import json
        try:
            analysis = json.loads(result)
            return analysis
        except json.JSONDecodeError:
            return {'level': '중급', 'follow_up_questions': []}
        
    except Exception as e:
        logger.error(f"질문 수준 분석 실패: {e}")
        return {'level': '중급', 'follow_up_questions': []}

def _enhance_answer_with_follow_ups(original_answer: str, follow_up_questions: List[str], 
                                   search_results: List[Dict], user_info: Dict[str, str], 
                                   openai_api_key: str = None) -> str:
    """
    기초 질문에 대해 예상 후속 질문들을 미리 답변하여 답변을 보강
    
    Args:
        original_answer: 원본 답변
        follow_up_questions: 예상 후속 질문 리스트
        search_results: 검색 결과
        user_info: 사용자 정보
        openai_api_key: OpenAI API 키
    
    Returns:
        보강된 답변
    """
    try:
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key or not follow_up_questions:
            return original_answer
        
        client = openai.OpenAI(api_key=api_key)
        
        # 검색 결과에서 관련 컨텍스트 추출
        context_text = ""
        for i, result in enumerate(search_results[:3], 1):
            text = result.get('text', '')
            if len(text) > 300:
                text = text[:300] + "..."
            context_text += f"[{i}] {text}\n\n"
        
        # 사용자 정보 반영
        user_context = ""
        if user_info.get('department'):
            user_context += f"사용자 부서: {user_info['department']}\n"
        if user_info.get('position'):
            user_context += f"사용자 직급: {user_info['position']}\n"
        
        system_prompt = f"""당신은 한국인터넷진흥원(KISA)의 업무 가이드 전문가입니다.
사용자가 기초적인 질문을 했을 때, 관련된 추가 정보를 자연스럽게 제공하고 유도형 질문으로 더 도움이 되는 응답을 제공하세요.

사용자 정보:
{user_context}

참고 컨텍스트:
{context_text}

답변 형식:
1. 질문에 대한 답변을 자연스럽게 제공
2. 관련된 추가 정보나 팁을 자연스럽게 포함
3. 유도형 질문으로 사용자의 관심을 끌어 추가 질문을 유도
4. 전체적으로 자연스럽고 유용한 정보 제공

한국어로 작성하고, 사용자 부서와 직급을 고려하여 적절한 어조로 답변하세요.
"원본 답변", "추가로 궁금할 수 있는 내용" 같은 키워드는 사용하지 말고 자연스럽게 작성하세요."""

        user_prompt = f"""현재 답변: {original_answer}

사용자가 추가로 궁금해할 수 있는 내용들:
{chr(10).join([f"- {q}" for q in follow_up_questions])}

위 내용들을 자연스럽게 포함하여 더 도움이 되는 답변으로 보강해주세요.
유도형 질문을 사용하여 사용자가 추가로 질문할 수 있도록 유도하세요."""

        response_result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=800
        )
        
        enhanced_answer = response_result.choices[0].message.content.strip()
        return enhanced_answer
        
    except Exception as e:
        logger.error(f"후속 질문 답변 보강 실패: {e}")
        return original_answer

def get_user_context(conversation_history: List[Dict]) -> Dict[str, str]:
    """
    대화 히스토리에서 사용자 정보 추출
    
    Args:
        conversation_history: 대화 히스토리
    
    Returns:
        사용자 정보 딕셔너리
    """
    if not conversation_history:
        return {}
    
    for msg in conversation_history:
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content", "")
            if "user_context:" in content:
                try:
                    import json
                    context_str = content.split("user_context:")[1].strip()
                    user_info = json.loads(context_str)
                    print(f"DEBUG: 사용자 정보 추출 성공: {user_info}")
                    return user_info
                except Exception as e:
                    print(f"DEBUG: 사용자 정보 추출 실패: {e}")
                    return {}
    
    print(f"DEBUG: 대화 히스토리에서 사용자 정보를 찾을 수 없음. 히스토리 길이: {len(conversation_history)}")
    return {}

def answer_query(query: str, openai_api_key: str = None, explicit_domain: str = None, conversation_history: List[Dict] = None) -> Dict[str, Any]:
    """
    질문에 대한 완전한 RAG 답변 생성 (멀티턴 대화 지원)
    
    Args:
        query: 사용자 질문
        openai_api_key: OpenAI API 키 (선택사항)
        explicit_domain: 명시적 도메인 (선택사항)
        conversation_history: 대화 히스토리 (선택사항)
    
    Returns:
        답변, 메타데이터, 참고문서를 포함한 딕셔너리
    """
    start_time = time.time()
    
    try:
        logger.info(f"RAG 파이프라인 시작 - 질문: {query}")
        print(f"DEBUG: RAG 파이프라인 시작 - 질문: {query}")
        
        # 👤 사용자 입력 종합 분석 (한 번의 LLM 호출로 모든 정보 추출)
        input_analysis = analyze_user_input(query, openai_api_key)
        logger.info(f"👤 사용자 입력 분석: {input_analysis}")
        print(f"DEBUG: 👤 사용자 입력 분석: {input_analysis}")
        
        # 사용자 정보가 있으면 대화 컨텍스트에 저장
        user_info = input_analysis.get('user_info')
        if user_info and any(user_info.values()):
            conversation_history = update_user_context(conversation_history, user_info)
            logger.info(f"👤 사용자 정보 저장됨: {user_info}")
            print(f"DEBUG: 👤 사용자 정보 저장됨: {user_info}")
        
        # 기존 사용자 정보 가져오기
        existing_user_info = get_user_context(conversation_history)
        logger.info(f"👤 기존 사용자 정보: {existing_user_info}")
        print(f"DEBUG: 👤 기존 사용자 정보: {existing_user_info}")
        
        # 🏢 부서 소개 질문 처리 (기존 사용자 정보 고려)
        is_department_intro = input_analysis.get('is_department_intro', False)
        department = input_analysis.get('department')
        
        # 기존 사용자 정보에서 부서 정보 가져오기
        existing_department = existing_user_info.get('department')
        
        # 부서 소개 질문이거나 "우리부서" 같은 표현이 있는 경우
        if (is_department_intro and department) or ('우리부서' in query or '우리팀' in query or '우리 조직' in query):
            # 현재 질문에서 부서를 찾지 못했지만 기존 정보가 있으면 사용
            if not department and existing_department:
                department = existing_department
                is_department_intro = True
                logger.info(f"🏢 기존 사용자 정보에서 부서 추출: {department}")
                print(f"DEBUG: 🏢 기존 사용자 정보에서 부서 추출: {department}")
        
        # 멀티턴 대화에서 "내가 해야 할 일" 같은 질문 처리
        if not is_department_intro and not department and existing_department:
            # 기존 대화에서 부서 정보가 있고, 현재 질문이 업무 관련이면 부서 소개로 처리
            work_related_keywords = ['해야할', '해야 할', '업무', '일', '뭘', '무슨', '어떤', '해야돼', '해야 돼']
            if any(keyword in query for keyword in work_related_keywords):
                department = existing_department
                is_department_intro = True
                logger.info(f"🏢 멀티턴 대화에서 부서 정보 활용: {department}")
                print(f"DEBUG: 🏢 멀티턴 대화에서 부서 정보 활용: {department}")
        
        if is_department_intro and department:
            logger.info(f"🏢 부서 소개 질문 감지: {department}")
            print(f"DEBUG: 🏢 부서 소개 질문 감지: {department}")
            
            # 부서별 업무 소개를 위한 동적 응답 생성
            try:
                api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
                if api_key:
                    client = openai.OpenAI(api_key=api_key)
                    
                    # LLM이 동적으로 질문 유형을 판단하고 응답 생성
                    system_prompt = f"""당신은 한국인터넷진흥원(KISA)의 업무 가이드 전문가입니다.
사용자의 질문을 분석하여 "{department}"에 맞는 적절한 답변을 제공해주세요.

사용자 질문 분석 및 응답 가이드:
1. 질문 유형 파악:
   - 업무 관련: "업무 알려줘", "뭘 해야해", "업무가 궁금해"
   - 학습 관련: "공부", "학습", "배우", "스킬", "자격증"
   - 도구/기술 관련: "도구", "기술", "프로그램", "시스템"
   - 기타: 조직문화, 커리어, 성장 등

2. 답변 구성:
   - 질문 유형에 맞는 구체적이고 실용적인 정보 제공
   - 해당 부서의 특성과 업무 환경 고려
   - 실무에 바로 적용 가능한 조언 포함
   - 추가 학습이나 발전 방향 제시

3. 답변 스타일:
   - 친근하고 전문적인 어조
   - 구체적인 예시와 설명 포함
   - 단계별 가이드나 체크리스트 제공
   - 추가 질문을 유도하는 마무리

한국어로 작성하세요."""
                    
                    # LLM이 질문을 분석하여 적절한 답변 생성
                    user_prompt = f"""사용자 질문: "{query}"
사용자 부서: "{department}"

위 질문을 분석하여 {department}에 맞는 적절한 답변을 제공해주세요.
질문의 의도와 맥락을 파악하고, 실용적이고 구체적인 정보를 포함하여 답변하세요."""
                    
                    response_result = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )
                    
                    department_response = response_result.choices[0].message.content.strip()
                    
                    return {
                        'answer': department_response,
                        'metadata': {
                            'processing_time': time.time() - start_time,
                            'query_type': 'department_intro',
                            'department': department,
                            'user_info': user_info
                        },
                        'references': []
                    }
            except Exception as e:
                logger.error(f"부서 소개 응답 생성 실패: {e}")
        
        # 🚀 간단한 질문 감지 (이미 분석된 결과 사용)
        is_simple_query = input_analysis.get('is_simple_greeting', True)
        
        if conversation_history and len(conversation_history) > 0:
            logger.info(f"💬 대화 히스토리 감지 ({len(conversation_history)}개 메시지), 멀티턴 대화로 처리")
            print(f"DEBUG: 대화 히스토리 감지, 멀티턴 대화로 처리")
            if not is_simple_query:
                logger.info("복잡한 질문으로 분류되어 RAG 파이프라인 실행")
                print(f"DEBUG: 복잡한 질문으로 분류되어 RAG 파이프라인 실행")
        
        if is_simple_query:
            logger.info(f"⚡ 간단한 질문 감지, LLM으로 직접 응답 생성: {query}")
            print(f"DEBUG: ⚡ 간단한 질문 감지, LLM으로 직접 응답 생성")
            
            # LLM에게 간단한 응답 생성 요청
            try:
                api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
                if not api_key:
                    # API 키가 없는 경우에만 기본 응답 사용
                    response = "안녕하세요! 업무 관련 궁금한 사항이 있으시면 언제든 문의해 주세요."
                else:
                    client = openai.OpenAI(api_key=api_key)
                    
                    # 대화 히스토리를 고려한 간단한 응답 생성
                    messages = []
                    
                    # 시스템 프롬프트 추가
                    system_prompt = """당신은 한국인터넷진흥원(KISA)의 업무 가이드 챗봇입니다.
사용자의 간단한 인사말이나 짧은 대화에 친근하고 전문적으로 응답하세요.
자기소개가 포함된 경우 해당 정보를 인정하고 반응하세요.
업무 관련 도움을 제공할 준비가 되어 있음을 알려주세요.
2-3문장으로 간결하게 작성하세요."""
                    
                    messages.append({"role": "system", "content": system_prompt})
                    
                    # 대화 히스토리가 있으면 추가
                    if conversation_history and len(conversation_history) > 0:
                        for msg in conversation_history:
                            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                                role = msg.get("role", "user")
                                if role in ["user", "assistant"]:
                                    content = str(msg.get("content", ""))[:1000]  # 길이 제한
                                    if content.strip():
                                        messages.append({"role": role, "content": content})
                    
                    # 현재 사용자 입력 추가
                    messages.append({"role": "user", "content": query})
                    
                    response_result = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=150
                    )
                    
                    response = response_result.choices[0].message.content.strip()
                    
            except Exception as e:
                logger.error(f"간단한 응답 생성 실패: {e}")
                # 오류 시 기본 응답 사용
                response = "안녕하세요! 업무 관련 궁금한 사항이 있으시면 언제든 문의해 주세요."
            
            total_time = time.time() - start_time
            logger.info(f"⚡ 간단한 질문 처리 완료 (총 소요시간: {total_time:.2f}초)")
            
            return {
                'answer': response,
                'contexts': [],
                'sources': [],
                'keywords': [],
                'domains': [],
                'search_strategy': 'simple_response',
                'total_time': total_time,
                'search_time': 0,
                'answer_time': total_time
            }
        
        # 🔥 복잡한 질문 처리 (통합된 system_prompt가 보안 검증 포함)
        logger.info("복잡한 질문 감지, 전체 RAG 파이프라인 실행")
        print(f"DEBUG: 복잡한 질문 감지, 전체 RAG 파이프라인 실행")
        
        # 1단계: 키워드 추출
        logger.info("1단계: 키워드 추출 시작")
        keywords_start = time.time()
        try:
            keywords = extract_keywords(query, openai_api_key)
            logger.info(f"키워드 추출 완료 (소요시간: {time.time() - keywords_start:.2f}초)")
            print(f"DEBUG: 추출된 키워드: {keywords}")
        except Exception as e:
            logger.error(f"키워드 추출 실패: {e}")
            keywords = []
            print(f"WARNING: 키워드 추출 실패, 빈 리스트 사용: {e}")
        
        # 2단계: 도메인 추정
        logger.info("2단계: 도메인 추정 시작")
        domain_start = time.time()
        try:
            if explicit_domain:
                # 명시적으로 지정된 도메인 우선
                estimated_domains = [explicit_domain]
                logger.info(f"명시적 도메인 사용: {explicit_domain}")
                print(f"DEBUG: 명시적 도메인 사용: {explicit_domain}")
            else:
                # 키워드 기반 도메인 추정
                estimated_domains = guess_domains_from_keywords(keywords)
                logger.info(f"도메인 추정 완료 (소요시간: {time.time() - domain_start:.2f}초)")
                print(f"DEBUG: 추정된 도메인: {estimated_domains}")
        except Exception as e:
            logger.error(f"도메인 추정 실패: {e}")
            estimated_domains = []
            print(f"WARNING: 도메인 추정 실패, 빈 리스트 사용: {e}")
        
        # 3단계: 향상된 RAG 검색 (새로운 메타데이터 구조 활용)
        logger.info("3단계: RAG 검색 시작")
        search_start = time.time()
        
        try:
            # 검색 전략 결정
            search_strategy = _determine_search_strategy(query, keywords, estimated_domains)
            logger.info(f"검색 전략 결정: {search_strategy}")
            print(f"DEBUG: 검색 전략: {search_strategy}")
            
            # 전략에 따른 검색 실행
            if search_strategy['type'] == 'form_specific':
                # 서식 전용 검색
                searcher = RagSearcher()
                search_results = searcher.search_forms(query=query, top_k=10)
                logger.info(f"서식 전용 검색 실행 - 결과 수: {len(search_results)}")
                print(f"DEBUG: 서식 전용 검색 실행 - 결과 수: {len(search_results)}")
            elif search_strategy['type'] == 'domain_specific':
                search_results = RagSearcher().search_by_domain(
                    query=query, 
                    domain=search_strategy['domain'], 
                    top_k=10
                )
            elif search_strategy['type'] == 'file_type_specific':
                search_results = RagSearcher().search_by_file_type(
                    query=query, 
                    file_type=search_strategy['file_type'], 
                    top_k=10
                )
            elif search_strategy['type'] == 'recency_aware':
                search_results = RagSearcher().search_by_recency(
                    query=query, 
                    min_recency=search_strategy['min_recency'], 
                    top_k=10
                )
            else:
                # 하이브리드 검색 (기본)
                searcher = RagSearcher()
                search_results = searcher.hybrid_search(
                    query=query,
                    domain_list=estimated_domains if estimated_domains else None,
                    file_types=search_strategy.get('file_types'),
                    min_recency=search_strategy.get('min_recency'),
                    top_k=10
                )
            
            # 사용자 부서에 맞게 검색 결과 우선순위 조정
            user_department = existing_user_info.get('department', '')
            if user_department:
                search_results = prioritize_results_by_department(search_results, user_department, openai_api_key)
                logger.info(f"부서별 우선순위 조정 적용: {user_department}")
                print(f"DEBUG: 부서별 우선순위 조정 적용: {user_department}")
            
            logger.info(f"검색 완료 (소요시간: {time.time() - search_start:.2f}초, 결과 수: {len(search_results)})")
            print(f"DEBUG: 검색 결과 수: {len(search_results)}")
            
        except Exception as e:
            logger.error(f"RAG 검색 실패: {e}")
            search_results = []
            print(f"ERROR: RAG 검색 실패: {e}")
        
        # 4단계: 답변 생성
        logger.info("4단계: 답변 생성 시작")
        answer_start = time.time()
        
        try:
            if search_results:
                # 검색 결과가 3개 미만인 경우 사용자 친화적 메시지 생성
                if len(search_results) < 3:
                    result = {
                        'success': True,
                        'answer': "질문을 조금 더 명확하게 해주시면 감사합니다.",
                        'used_domains': estimated_domains,
                        'search_strategy': search_strategy,
                        'top_docs': [],
                        'sources': []
                    }
                    logger.info("검색 결과 부족으로 기본 메시지 반환")
                    return result
                
                # 서식 검색 결과인 경우 특별 처리
                if search_strategy['type'] == 'form_specific':
                    answer = _generate_form_response(query, search_results[:5])
                else:
                    # 일반 답변 생성
                    contexts = [result['text'] for result in search_results[:5]]
                    
                    # 시스템 및 사용자 프롬프트 로드
                    system_prompt, user_prompt = _init_prompts()
                    
                    # 답변 생성 (올바른 인자로 호출)
                    answer = make_answer(
                        query=query,
                        contexts=search_results[:5],  # 전체 결과 객체 전달
                        api_key=None,  # 환경변수에서 자동으로 가져옴
                        conversation_history=conversation_history, # 대화 히스토리 전달
                        user_info=existing_user_info  # 사용자 정보 전달
                    )
                
                # 답변 품질 검증
                if not validate_answer_quality(answer, query):
                    logger.warning("답변 품질이 낮습니다. 기본 메시지로 대체합니다.")
                    print("WARNING: 답변 품질이 낮습니다. 기본 메시지로 대체합니다.")
                    answer = "죄송합니다. 질문에 대한 적절한 답변을 생성하지 못했습니다. 다른 방식으로 질문해 주시거나, 관련 도메인을 명시해 주세요."
                
                # 질문 수준 분석 및 예상 후속 질문 미리 답변
                try:
                    question_analysis = analyze_question_level(query, openai_api_key)
                    question_level = question_analysis.get('level', '중급')
                    follow_up_questions = question_analysis.get('follow_up_questions', [])
                    
                    logger.info(f"질문 수준 분석: {question_level}, 예상 후속 질문: {len(follow_up_questions)}개")
                    print(f"DEBUG: 질문 수준 분석: {question_level}, 예상 후속 질문: {len(follow_up_questions)}개")
                    
                    # 기초 수준 질문의 경우 예상 후속 질문들을 미리 답변
                    if question_level == '기초' and follow_up_questions:
                        enhanced_answer = _enhance_answer_with_follow_ups(
                            answer, follow_up_questions, search_results[:3], 
                            existing_user_info, openai_api_key
                        )
                        if enhanced_answer:
                            answer = enhanced_answer
                            logger.info("기초 질문에 대한 예상 후속 질문 답변 추가")
                            print("DEBUG: 기초 질문에 대한 예상 후속 질문 답변 추가")
                    
                except Exception as e:
                    logger.error(f"질문 수준 분석 실패: {e}")
                    print(f"WARNING: 질문 수준 분석 실패: {e}")
                
                # 참고 문서 정보 생성 (새로운 메타데이터 활용)
                if search_strategy['type'] == 'form_specific':
                    # 서식 검색 결과의 경우 서식 정보를 소스로 제공
                    sources = _format_form_sources(search_results[:5])
                else:
                    sources = _format_sources_with_metadata(search_results[:5])
                
                result = {
                    'success': True,
                    'answer': answer,
                    'used_domains': estimated_domains,
                    'search_strategy': search_strategy,
                    'top_docs': search_results[:5],
                    'sources': sources
                }
                
                logger.info(f"답변 생성 완료 (소요시간: {time.time() - answer_start:.2f}초)")
                
            else:
                # 검색 결과가 없는 경우
                result = {
                    'success': True,
                    'answer': "질문을 조금 더 명확하게 해주시면 감사합니다.",
                    'used_domains': estimated_domains,
                    'search_strategy': search_strategy,
                    'top_docs': [],
                    'sources': []
                }
                logger.info("검색 결과 없음으로 기본 메시지 반환")
            
            total_time = time.time() - start_time
            logger.info(f"RAG 파이프라인 완료 (총 소요시간: {total_time:.2f}초)")
            
            return result
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            print(f"ERROR: 답변 생성 실패: {e}")
            raise
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"RAG 파이프라인 오류 (소요시간: {total_time:.2f}초): {e}")
        print(f"RAG 파이프라인 오류: {e}")
        return {
            'success': False,
            'error': str(e),
            'answer': "죄송합니다. 시스템 오류가 발생했습니다."
        }

def _is_form_related_query(query: str, keywords: List[str]) -> bool:
    """
    서식 관련 질문인지 판단
    
    Args:
        query: 사용자 질문
        keywords: 추출된 키워드
    
    Returns:
        서식 관련 질문 여부
    """
    query_lower = query.lower()
    
    # 서식 관련 키워드 패턴
    form_keywords = [
        '서식', '양식', '신청서', '제출서', '청구서', '요청서', '보고서', '평가서',
        '확인서', '서약서', '계약서', '승인서', '통지서', '등록서', '변경서',
        '관리서', '운영서', '처리서', '대장', '접수증', '일지', '체크리스트',
        '점검표', '결과표', '검토서', '완료확인서', '취소신청서', '재발급신청서',
        '인증연장신청서', '윤리서약서', '보안서약서', '직무윤리서약서'
    ]
    
    # 질문에 서식 관련 키워드가 포함되어 있는지 확인
    for keyword in form_keywords:
        if keyword in query_lower:
            return True
    
    # 키워드 리스트에서도 확인
    for keyword in keywords:
        if any(form_kw in keyword.lower() for form_kw in form_keywords):
            return True
    
    # 서식 요청 패턴 확인
    form_request_patterns = [
        '서식 주세요', '양식 주세요', '신청서 주세요', '양식 찾아줘',
        '서식 찾아줘', '신청서 찾아줘', '양식 다운로드', '서식 다운로드',
        '어떤 서식', '어떤 양식', '필요한 서식', '필요한 양식'
    ]
    
    for pattern in form_request_patterns:
        if pattern in query_lower:
            return True
    
    return False

def _determine_search_strategy(query: str, keywords: List[str], estimated_domains: List[str]) -> Dict[str, Any]:
    """
    질문과 키워드를 분석하여 최적의 검색 전략 결정
    
    Args:
        query: 사용자 질문
        keywords: 추출된 키워드
        estimated_domains: 추정된 도메인
    
    Returns:
        검색 전략 딕셔너리
    """
    query_lower = query.lower()
    
    # 0. 서식 관련 질문 우선 검사
    if _is_form_related_query(query, keywords):
        return {
            'type': 'form_specific',
            'confidence': 'high',
            'priority': 'forms_first'
        }
    
    # 1. 도메인 특정 검색 전략
    if estimated_domains and len(estimated_domains) == 1:
        return {
            'type': 'domain_specific',
            'domain': estimated_domains[0],
            'confidence': 'high'
        }
    
    # 2. 문서 타입 특정 검색 전략
    doc_type_keywords = {
        '정관': ['정관', '기본법', '조직법'],
        '규정': ['규정', '운영규정', '관리규정'],
        '규칙': ['규칙', '세부규칙', '실행규칙'],
        '지침': ['지침', '업무지침', '운영지침']
    }
    
    for doc_type, type_keywords in doc_type_keywords.items():
        if any(keyword in query_lower for keyword in type_keywords):
            return {
                'type': 'file_type_specific',
                'file_type': doc_type,
                'confidence': 'medium'
            }
    
    # 3. 최신성 인식 검색 전략
    recency_keywords = ['최신', '최근', '새로운', '업데이트', '변경', '수정']
    if any(keyword in query_lower for keyword in recency_keywords):
        return {
            'type': 'recency_aware',
            'min_recency': 2,  # 최신성 점수 2 이상
            'confidence': 'medium'
        }
    
    # 4. 복합 검색 전략 (기본)
    return {
        'type': 'hybrid',
        'domain_list': estimated_domains,
        'file_types': None,
        'min_recency': None,
        'confidence': 'low'
    }

def _generate_form_response(query: str, form_results: List[Dict[str, Any]]) -> str:
    """
    서식 검색 결과를 기반으로 서식 제공 응답 생성
    
    Args:
        query: 사용자 질문
        form_results: 서식 검색 결과
    
    Returns:
        서식 제공 응답
    """
    if not form_results:
        return "죄송합니다. 요청하신 서식을 찾을 수 없습니다. 다른 키워드로 검색해 보시거나 관련 부서에 문의해 주세요."
    
    response_parts = []
    
    # 서식 목록 구성
    form_list = []
    for i, result in enumerate(form_results, 1):
        form_title = result.get('form_title', '')
        form_file_uri = result.get('form_file_uri', '')
        source_file = result.get('file_name', '')
        page = result.get('pages', '')
        
        form_info = f"{i}. {form_title}"
        if source_file:
            form_info += f" (출처: {source_file}"
            if page:
                form_info += f", p.{page}"
            form_info += ")"
        
        form_list.append(form_info)
        
        # S3 파일 링크가 있으면 추가
        if form_file_uri:
            # S3 키 추출 (s3://bucket/key 형식에서 key 부분만)
            s3_key = form_file_uri.replace('s3://companypolicy/', '')
            # S3 퍼블릭 URL 직접 생성
            bucket_name = 'companypolicy'
            region = 'ap-northeast-2'
            download_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
            # 파일명 추출 (S3 키에서 마지막 부분)
            filename = s3_key.split('/')[-1]
            # 클릭 가능한 마크다운 링크 형식으로 변경
            form_list.append(f"  ({download_url})")
    
    # 응답 구성
    response_parts.append("요청하신 서식을 찾았습니다:")
    response_parts.append("")
    response_parts.extend(form_list)
    response_parts.append("")
    response_parts.append("💡 서식 사용 시 주의사항:")
    response_parts.append("- 서식은 최신 버전인지 확인해 주세요")
    response_parts.append("- 작성 전 관련 규정을 숙지해 주세요")
    response_parts.append("- 제출 전 내용을 다시 한번 검토해 주세요")
    
    return "\n".join(response_parts)

def _format_form_sources(form_results: List[Dict[str, Any]]) -> List[str]:
    """
    서식 검색 결과를 소스 정보로 포맷팅
    
    Args:
        form_results: 서식 검색 결과 리스트
    
    Returns:
        포맷팅된 서식 소스 리스트
    """
    sources = []
    
    for result in form_results:
        form_title = result.get('form_title', '')
        source_file = result.get('file_name', '')
        page = result.get('pages', '')
        form_file_uri = result.get('form_file_uri', '')
        
        source_info = f"서식: {form_title}"
        if source_file:
            source_info += f" (출처: {source_file}"
            if page:
                source_info += f", p.{page}"
            source_info += ")"
        
        if form_file_uri:
            source_info += f" [다운로드 가능]"
        
        sources.append(source_info)
    
    return sources

def _format_sources_with_metadata(search_results: List[Dict[str, Any]]) -> List[str]:
    """
    새로운 메타데이터를 활용하여 참고 문서 정보 포맷팅
    
    Args:
        search_results: 검색 결과 리스트
    
    Returns:
        포맷팅된 참고 문서 리스트
    """
    sources = []
    
    for result in search_results:
        # 파일명에서 날짜 패턴 제거
        file_name = result.get('file_name', '')
        if file_name:
            # 날짜 패턴 (YYYYMMDD) 제거
            file_name = re.sub(r'\(\d{6}\)', '', file_name).strip()
            file_name = file_name.replace('_', ' ')
        
        # 페이지 정보
        page_info = result.get('pages', '')
        if page_info:
            page_str = f"p.{page_info}"
        else:
            page_str = ""
        
        # 도메인 정보
        domain_info = result.get('domain_primary', '')
        if domain_info and domain_info != '일반':
            domain_str = f" ({domain_info})"
        else:
            domain_str = ""
        
        # 최신성 정보
        recency_score = result.get('recency_score', 1)
        if recency_score >= 3:
            recency_str = " [최신]"
        elif recency_score >= 2:
            recency_str = " [최근]"
        else:
            recency_str = ""
        
        # 최종 소스 문자열 조합
        source_parts = [part for part in [file_name, page_str, domain_str, recency_str] if part]
        source_str = " ".join(source_parts).strip()
        
        if source_str:
            sources.append(source_str)
    
    return sources

def quick_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    빠른 검색 (답변 생성 없이)
    
    Args:
        query: 검색 질문
        top_k: 반환할 결과 수
    
    Returns:
        검색 결과 리스트
    """
    try:
        searcher = RagSearcher()
        results = searcher.search(query, top_k=top_k)
        return results
    except Exception as e:
        logger.error(f"빠른 검색 오류: {e}")
        print(f"빠른 검색 오류: {e}")
        return []

def get_domain_suggestions(query: str) -> List[str]:
    """
    질문에 대한 도메인 제안
    
    Args:
        query: 사용자 질문
    
    Returns:
        제안 도메인 리스트
    """
    try:
        keywords = extract_keywords(query)
        suggestions = guess_domains_from_keywords(keywords)
        return suggestions
    except Exception as e:
        logger.error(f"도메인 제안 오류: {e}")
        print(f"도메인 제안 오류: {e}")
        return []

def health_check() -> Dict[str, Any]:
    """
    RAG 시스템 상태 확인
    
    Returns:
        시스템 상태 정보
    """
    try:
        searcher = RagSearcher()
        
        # Qdrant 연결 상태 확인
        qdrant_health = searcher.health_check()
        
        # 컬렉션 정보 조회
        collection_info = searcher.get_collection_info()
        
        # 키워드 추출 테스트
        keyword_test = extract_keywords("테스트 질문")
        
        return {
            'status': 'healthy' if all([qdrant_health, collection_info, keyword_test]) else 'degraded',
            'qdrant_connection': qdrant_health,
            'collection_info': collection_info,
            'keyword_extraction': bool(keyword_test),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"상태 확인 오류: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.datetime.now().isoformat()
        }

def rag_answer_enhanced(user_query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
    """
    향상된 RAG 답변 생성 (멀티턴 대화 지원)
    
    Args:
        user_query: 사용자 질문
        conversation_history: 대화 히스토리 (선택사항)
    
    Returns:
        답변과 메타데이터를 포함한 결과
    """
    try:
        # OpenAI API 키는 환경변수에서 자동으로 가져옴
        result = answer_query(user_query, conversation_history=conversation_history)
        
        # answer_query는 항상 answer를 반환하므로 success 체크 불필요
        search_strategy = result.get('search_strategy', '')
        
        return {
            'answer': result.get('answer', '죄송합니다. 답변을 생성할 수 없습니다.'),
            'sources': result.get('sources', []),
            'rag_used': search_strategy != 'simple_response',  # 간단한 응답이 아니면 RAG 사용
            'metadata': {
                'domains': result.get('domains', []),
                'search_strategy': search_strategy,
                'keywords': result.get('keywords', []),
                'total_time': result.get('total_time', 0),
                'search_time': result.get('search_time', 0),
                'answer_time': result.get('answer_time', 0),
                'conversation_history_used': bool(conversation_history),
                'user_info': result.get('metadata', {}).get('user_info')  # 사용자 정보 포함
            }
        }
            
    except Exception as e:
        logger.error(f"향상된 RAG 답변 생성 오류: {e}")
        print(f"향상된 RAG 답변 생성 오류: {e}")
        return {
            'answer': '죄송합니다. 시스템 오류가 발생했습니다.',
            'sources': [],
            'rag_used': False,
            'metadata': {'error': str(e)}
        }