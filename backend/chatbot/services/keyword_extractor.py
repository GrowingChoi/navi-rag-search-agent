"""
키워드 추출기
기존 OpenAI SDK 방식을 유지하면서 키워드 추출 기능을 제공합니다.
"""

import json
import re
import os
from typing import List, Optional
from django.conf import settings

def extract_keywords_openai(query: str, api_key: Optional[str] = None) -> List[str]:
    """
    OpenAI를 사용하여 질문에서 핵심 키워드 추출
    
    Args:
        query: 사용자 질문
        api_key: OpenAI API 키 (None이면 settings에서 가져옴)
    
    Returns:
        추출된 키워드 리스트
    """
    try:
        # OpenAI SDK import (기존 방식 유지)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key or settings.OPENAI_API_KEY)
            model = "gpt-4o-mini"
        except ImportError:
            # 구버전 OpenAI SDK 지원
            import openai
            openai.api_key = api_key or settings.OPENAI_API_KEY
            model = "gpt-4o-mini"
        
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

        # 시스템 프롬프트 로드
        try:
            system_prompt_path = '/app/config/system_prompt.md'
            system_prompt = load_prompt(system_prompt_path, 
                                       default="규정/지침/규칙 문서 질의를 위한 핵심 키워드를 최대 5~8개 JSON 배열로만 출력하세요.")
        except FileNotFoundError:
            system_prompt = "규정/지침/규칙 문서 질의를 위한 핵심 키워드를 최대 5~8개 JSON 배열로만 출력하세요."
            print("WARNING: system_prompt.md not found, using default prompt")

        user_prompt = f"질문: {query}\n\n키워드 배열:"

        try:
            # OpenAI 1.x 방식
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                max_tokens=100,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            content = response.choices[0].message.content.strip()
        except AttributeError:
            # OpenAI 0.x 방식
            response = openai.ChatCompletion.create(
                model=model,
                temperature=0.2,
                max_tokens=100,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            content = response.choices[0].message.content.strip()
        
        # JSON 파싱 시도
        try:
            # JSON 배열 추출
            json_match = re.search(r'\[.*?\]', content)
            if json_match:
                keywords = json.loads(json_match.group())
                if isinstance(keywords, list):
                    return [str(kw).strip() for kw in keywords if kw.strip()]
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # JSON 파싱 실패 시 텍스트에서 키워드 추출
        return extract_keywords_fallback(query)
        
    except Exception as e:
        print(f"OpenAI 키워드 추출 실패: {e}")
        return extract_keywords_fallback(query)

def extract_keywords_fallback(query: str) -> List[str]:
    """
    OpenAI 실패 시 사용하는 fallback 키워드 추출
    
    Args:
        query: 사용자 질문
    
    Returns:
        추출된 키워드 리스트
    """
    # 한국어 키워드 추출 (2자 이상)
    keywords = re.findall(r'[가-힣]{2,}', query)
    
    # 불용어 제거
    stopwords = ['이', '가', '을', '를', '에', '에서', '로', '으로', '의', '와', '과', '도', '는', '은', '어떻게', '무엇', '언제', '어디서', '왜', '어떤', '몇', '얼마나']
    keywords = [kw for kw in keywords if kw not in stopwords]
    
    # 상위 5개 키워드 반환
    return keywords[:5]

def extract_keywords(query: str, api_key: Optional[str] = None) -> List[str]:
    """
    키워드 추출 메인 함수
    
    Args:
        query: 사용자 질문
        api_key: OpenAI API 키 (선택사항)
    
    Returns:
        추출된 키워드 리스트
    """
    # OpenAI API 키가 있으면 OpenAI 사용, 없으면 fallback
    if api_key or hasattr(settings, 'OPENAI_API_KEY'):
        return extract_keywords_openai(query, api_key)
    else:
        return extract_keywords_fallback(query) 