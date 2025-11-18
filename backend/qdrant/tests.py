from django.test import TestCase
from django.conf import settings
from .models import DocumentVector


class DocumentVectorModelTest(TestCase):
    """DocumentVector 모델 테스트"""
    
    def test_document_vector_creation(self):
        """DocumentVector 모델 생성 테스트"""
        document = DocumentVector.objects.create(
            document_name="테스트 문서",
            document_path="/test/path/document.pdf",
            vector_id="test_doc_001"
        )
        
        self.assertEqual(document.document_name, "테스트 문서")
        self.assertEqual(document.document_path, "/test/path/document.pdf")
        self.assertEqual(document.vector_id, "test_doc_001")
        self.assertIsNotNone(document.created_at)
        self.assertIsNotNone(document.updated_at)
    
    def test_document_vector_string_representation(self):
        """DocumentVector 모델 문자열 표현 테스트"""
        document = DocumentVector.objects.create(
            document_name="테스트 문서",
            document_path="/test/path/document.pdf",
            vector_id="test_doc_001"
        )
        
        expected_string = "테스트 문서 - test_doc_001"
        self.assertEqual(str(document), expected_string)
