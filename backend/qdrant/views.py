from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
import os
from .services import QdrantService
import logging

logger = logging.getLogger(__name__)


@api_view(['POST'])
def add_document(request):
    """PDF 문서를 벡터화하여 Qdrant에 추가"""
    try:
        document_name = request.data.get('document_name')
        pdf_path = request.data.get('pdf_path')
        batch_size = request.data.get('batch_size', 100)

        if not document_name or not pdf_path:
            return Response(
                {'error': 'document_name과 pdf_path가 필요합니다.'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        # 전체 경로 구성
        full_pdf_path = os.path.join(settings.BASE_DIR, '..', 'documents', 'kisa_pdf', pdf_path)

        if not os.path.exists(full_pdf_path):
            return Response(
                {'error': f'PDF 파일을 찾을 수 없습니다: {pdf_path}'}, 
                status=status.HTTP_404_NOT_FOUND
            )

        # Qdrant 서비스 초기화
        qdrant_service = QdrantService()

        # 컬렉션 생성 (없는 경우)
        qdrant_service.create_collection()

        # 문서 추가
        success = qdrant_service.add_document(full_pdf_path, document_name, batch_size)

        if success:
            return Response({
                'message': f'문서 "{document_name}" 추가 완료',
                'pdf_path': pdf_path,
                'batch_size': batch_size
            }, status=status.HTTP_201_CREATED)
        else:
            return Response({
                'error': '문서 추가 실패'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except Exception as e:
        logger.error(f"문서 추가 중 오류: {e}")
        return Response({
            'error': f'서버 오류: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def search_documents(request):
    """유사한 문서 검색"""
    try:
        query = request.data.get('query')
        limit = request.data.get('limit', 5)
        category = request.data.get('category')  # 카테고리 필터링

        if not query:
            return Response(
                {'error': '검색 쿼리가 필요합니다.'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        # Qdrant 서비스 초기화
        qdrant_service = QdrantService()

        # 검색 수행
        results = qdrant_service.search_similar(query, limit, category)

        return Response({
            'query': query,
            'category_filter': category,
            'results': results,
            'total_results': len(results)
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"문서 검색 중 오류: {e}")
        return Response({
            'error': f'서버 오류: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def collection_info(request):
    """컬렉션 정보 조회"""
    try:
        # Qdrant 서비스 초기화
        qdrant_service = QdrantService()

        # 컬렉션 정보 조회
        info = qdrant_service.get_collection_info()

        if info:
            return Response(info, status=status.HTTP_200_OK)
        else:
            return Response({
                'error': '컬렉션 정보를 가져올 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)

    except Exception as e:
        logger.error(f"컬렉션 정보 조회 중 오류: {e}")
        return Response({
            'error': f'서버 오류: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_categories(request):
    """사용 가능한 카테고리 목록 조회"""
    try:
        # Qdrant 서비스 초기화
        qdrant_service = QdrantService()

        # 카테고리 목록 조회
        categories = qdrant_service.get_categories()

        return Response({
            'categories': categories,
            'total_categories': len(categories)
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"카테고리 목록 조회 중 오류: {e}")
        return Response({
            'error': f'서버 오류: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['DELETE'])
def delete_collection(request):
    """컬렉션 삭제"""
    try:
        # Qdrant 서비스 초기화
        qdrant_service = QdrantService()

        # 컬렉션 삭제
        success = qdrant_service.delete_collection()

        if success:
            return Response({
                'message': '컬렉션 삭제 완료'
            }, status=status.HTTP_200_OK)
        else:
            return Response({
                'error': '컬렉션 삭제 실패'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except Exception as e:
        logger.error(f"컬렉션 삭제 중 오류: {e}")
        return Response({
            'error': f'서버 오류: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def add_all_documents(request):
    """documents/kisa_pdf 폴더의 모든 PDF를 벡터화하여 추가"""
    try:
        batch_size = request.data.get('batch_size', 100)

        # Qdrant 서비스 초기화
        qdrant_service = QdrantService()

        # 컬렉션 생성
        qdrant_service.create_collection()

        # PDF 폴더 경로
        pdf_folder = os.path.join(settings.BASE_DIR, '..', 'documents', 'kisa_pdf')

        if not os.path.exists(pdf_folder):
            return Response({
                'error': 'PDF 폴더를 찾을 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)

        # PDF 파일 목록
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

        if not pdf_files:
            return Response({
                'error': 'PDF 파일이 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)

        # 각 PDF 파일 처리
        success_count = 0
        failed_files = []

        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            document_name = os.path.splitext(pdf_file)[0]

            try:
                success = qdrant_service.add_document(pdf_path, document_name, batch_size)
                if success:
                    success_count += 1
                else:
                    failed_files.append(pdf_file)
            except Exception as e:
                logger.error(f"파일 {pdf_file} 처리 실패: {e}")
                failed_files.append(pdf_file)

        return Response({
            'message': f'처리 완료: {success_count}개 성공, {len(failed_files)}개 실패',
            'success_count': success_count,
            'failed_files': failed_files,
            'total_files': len(pdf_files),
            'batch_size': batch_size
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"전체 문서 추가 중 오류: {e}")
        return Response({
            'error': f'서버 오류: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
