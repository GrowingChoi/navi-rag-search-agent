from django.db import models


class DocumentVector(models.Model):
    """문서 벡터 정보를 저장하는 모델"""
    
    # 기본 정보
    document_name = models.CharField(max_length=255, verbose_name="문서명")
    document_path = models.CharField(max_length=500, verbose_name="문서 경로")
    vector_id = models.CharField(max_length=100, unique=True, verbose_name="벡터 ID")
    
    # 메타데이터
    category = models.CharField(max_length=100, verbose_name="카테고리", blank=True)
    register_date = models.CharField(max_length=20, verbose_name="등록일", blank=True)
    doc_title = models.CharField(max_length=255, verbose_name="문서 제목", blank=True)
    
    # 시간 정보
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="생성일")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="수정일")
    
    class Meta:
        db_table = 'document_vectors'
        verbose_name = "문서 벡터"
        verbose_name_plural = "문서 벡터들"
        ordering = ['-created_at']
        
    def __str__(self):
        return f"{self.document_name} - {self.category} ({self.vector_id})"
