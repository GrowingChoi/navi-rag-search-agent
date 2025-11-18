#!/usr/bin/env python
"""
향상된 RAG 시스템 테스트 스크립트
Django 환경 없이도 테스트할 수 있도록 구성
"""

import os
import sys
import django

# Django 설정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

def test_enhanced_rag():
    """향상된 RAG 시스템 테스트"""
    try:
        from chatbot.services.pipeline import answer_query, health_check
        
        print("🔍 향상된 RAG 시스템 테스트 시작")
        print("=" * 50)
        
        # 1. 시스템 상태 확인
        print("1️⃣ 시스템 상태 확인...")
        health_info = health_check()
        print(f"   상태: {health_info.get('status', 'unknown')}")
        print(f"   Qdrant: {health_info.get('qdrant', {}).get('status', 'unknown')}")
        print(f"   검색기: {health_info.get('searcher', 'unknown')}")
        print(f"   OpenAI: {health_info.get('openai', 'unknown')}")
        
        # 2. 간단한 질문 테스트
        print("\n2️⃣ 간단한 질문 테스트...")
        test_questions = [
            "안녕하세요",
            "연차 규정은 어떻게 되나요?",
            "개인정보 보호에 대한 규정을 알려주세요"
        ]
        
        for question in test_questions:
            print(f"\n   질문: {question}")
            try:
                result = answer_query(question)
                if result['success']:
                    print(f"   ✅ 성공: {result['answer'][:100]}...")
                    print(f"   📚 사용된 도메인: {result.get('used_domains', [])}")
                    print(f"   🔍 검색된 문서 수: {result.get('search_count', 0)}")
                    print(f"   📊 품질 점수: {result.get('quality', {}).get('score', 'N/A')}")
                else:
                    print(f"   ❌ 실패: {result.get('error', '알 수 없는 오류')}")
            except Exception as e:
                print(f"   ❌ 예외 발생: {str(e)}")
        
        print("\n✅ 테스트 완료!")
        
    except ImportError as e:
        print(f"❌ 모듈 import 실패: {e}")
        print("   Django 환경 설정을 확인해주세요.")
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    test_enhanced_rag() 