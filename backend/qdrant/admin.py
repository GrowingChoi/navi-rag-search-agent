from django.contrib import admin
from .models import DocumentVector


@admin.register(DocumentVector)
class DocumentVectorAdmin(admin.ModelAdmin):
    """DocumentVector 모델 관리자"""
    list_display = ('document_name', 'category', 'register_date', 'doc_title', 'vector_id', 'created_at')
    list_filter = ('category', 'register_date', 'created_at', 'updated_at')
    search_fields = ('document_name', 'doc_title', 'vector_id', 'category')
    readonly_fields = ('created_at', 'updated_at')
    ordering = ('-created_at',)
    
    fieldsets = (
        ('문서 정보', {
            'fields': ('document_name', 'doc_title', 'document_path', 'vector_id')
        }),
        ('메타데이터', {
            'fields': ('category', 'register_date')
        }),
        ('시간 정보', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    list_per_page = 50
