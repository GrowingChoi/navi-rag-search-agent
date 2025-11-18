# chatbot/views.py
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.views import APIView
from rest_framework import generics, status, viewsets, permissions
from authapp.utils import verify_token, get_user_from_token
from authapp.decorators import require_auth
from .models import Conversation, ChatMessage, ChatReport
from .serializers import ConversationSerializer, ChatMessageSerializer, ChatQuerySerializer, ChatReportSerializer
from .services.rag_service import rag_answer
from .services.pipeline import rag_answer_enhanced
from django.http import JsonResponse, HttpResponse
import boto3
import os
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)

class ConversationListView(generics.ListAPIView):

    """
    대화방 목록 조회
    """
    authentication_classes = []  # 개발 단계에서는 인증 클래스 제거
    permission_classes = [AllowAny]  # 개발 단계에서는 AllowAny
    serializer_class = ConversationSerializer

    def get_queryset(self):
        # JWT 토큰에서 user_id 추출하여 필터링
        auth_header = self.request.headers.get('Authorization')
        user_id = None
        
        print(f"DEBUG: ConversationListView - Authorization 헤더: {auth_header}")
        
        if auth_header:
            try:
                token_type, token = auth_header.split(' ')
                print(f"DEBUG: ConversationListView - 토큰 타입: {token_type}, 토큰: {token[:20]}...")
                
                if token_type.lower() == 'bearer':
                    from authapp.utils import verify_token
                    payload = verify_token(token)
                    print(f"DEBUG: ConversationListView - 토큰 검증 결과: {payload}")
                    
                    if payload:
                        user_id = payload.get('user_id')
                        print(f"DEBUG: ConversationListView - JWT에서 추출한 user_id: {user_id}")
                    else:
                        print(f"DEBUG: ConversationListView - 토큰 검증 실패")
                else:
                    print(f"DEBUG: ConversationListView - 잘못된 토큰 타입: {token_type}")
            except Exception as e:
                print(f"DEBUG: ConversationListView - JWT 파싱 실패: {str(e)}")
                import traceback
                print(f"DEBUG: ConversationListView - 상세 오류: {traceback.format_exc()}")
        else:
            print(f"DEBUG: ConversationListView - Authorization 헤더가 없음")
        
        if user_id:
            # user_id로 필터링된 대화방만 반환
            queryset = Conversation.objects.filter(user_id=user_id).order_by('-updated_at')
            print(f"DEBUG: ConversationListView - user_id {user_id}로 필터링된 대화방 수: {queryset.count()}")
            return queryset
        else:
            # user_id가 없으면 빈 쿼리셋 반환
            print(f"DEBUG: ConversationListView - user_id가 없어 빈 결과 반환")
            return Conversation.objects.none()

class ConversationCreateView(generics.CreateAPIView):

    """
    새 대화방 생성
    """
    authentication_classes = []  # 개발 단계에서는 인증 클래스 제거
    permission_classes = [AllowAny]  # 개발 단계에서는 AllowAny
    serializer_class = ConversationSerializer

    def perform_create(self, serializer):
        # 디버깅을 위한 로그 추가
        print(f"DEBUG: ConversationCreateView.perform_create() 호출됨")
        print(f"DEBUG: request.data = {self.request.data}")
        print(f"DEBUG: request.user = {getattr(self.request, 'user', 'No user')}")
        
        # JWT 토큰이 있으면 user_id 사용, 없으면 요청 데이터에서 가져오기
        if hasattr(self.request, 'user') and self.request.user.is_authenticated:
            user_id = self.request.user.id
            print(f"DEBUG: 인증된 사용자에서 user_id = {user_id}")
            serializer.save(user_id=user_id)
        else:
            # 개발 단계에서는 요청 데이터에서 user_id 가져오기
            user_id = self.request.data.get('user_id')
            print(f"DEBUG: 요청 데이터에서 user_id = {user_id}")
            if user_id:
                serializer.save(user_id=user_id)
            else:
                print(f"DEBUG: user_id가 없어 기본값으로 저장")
                serializer.save()
        
        # 저장된 객체 확인
        saved_obj = serializer.instance
        print(f"DEBUG: 저장된 객체 = {saved_obj}")
        print(f"DEBUG: 저장된 객체 ID = {saved_obj.id}")
        print(f"DEBUG: 저장된 객체 user_id = {saved_obj.user_id}")

    def create(self, request, *args, **kwargs):
        response = super().create(request, *args, **kwargs)
        return Response({
            'success': True,
            'message': '대화방이 생성되었습니다',
            'data': response.data
        }, status=status.HTTP_201_CREATED)

class ConversationDeleteView(generics.DestroyAPIView):
    """
    개별 채팅 삭제 API
    - JWT 토큰 기반 사용자 인증
    - 본인의 대화기록만 삭제 가능
    - 연관된 채팅 메시지도 함께 삭제
    """
    authentication_classes = []
    permission_classes = [AllowAny]
    serializer_class = ConversationSerializer
    lookup_field = 'conversation_id'
    
    def delete(self, request, *args, **kwargs):
        conversation_id = kwargs.get('conversation_id')
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return Response(
                {'success': False, 'message': '인증 토큰이 필요합니다.'},
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        try:
            # JWT 토큰에서 user_id 추출
            token_type, token = auth_header.split(' ')
            if token_type.lower() != 'bearer':
                return Response(
                    {'success': False, 'message': '올바른 토큰 형식이 아닙니다.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            payload = verify_token(token)
            if not payload:
                return Response(
                    {'success': False, 'message': '유효하지 않은 토큰입니다.'},
                    status=status.HTTP_401_UNAUTHORIZED
                )
            
            user_id = payload.get('user_id')
            if not user_id:
                return Response(
                    {'success': False, 'message': '사용자 정보를 찾을 수 없습니다.'},
                    status=status.HTTP_401_UNAUTHORIZED
                )
            
            print(f"DEBUG: 삭제 요청 - conversation_id: {conversation_id}, user_id: {user_id}")
            
            # 🔒 보안 강화: user_id와 conversation_id를 모두 확인
            try:
                conversation = Conversation.objects.get(
                    id=conversation_id,
                    user_id=user_id  # 반드시 본인의 대화기록만 삭제 가능
                )
                print(f"DEBUG: 권한 확인 성공 - 사용자 {user_id}의 대화기록 {conversation_id}")
            except Conversation.DoesNotExist:
                print(f"DEBUG: 권한 확인 실패 - conversation_id: {conversation_id}, user_id: {user_id}")
                return Response(
                    {'success': False, 'message': '해당 대화기록을 찾을 수 없거나 삭제 권한이 없습니다.'},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # 연관된 채팅 메시지 삭제 (CASCADE 설정이지만 명시적으로 처리)
            chat_messages = ChatMessage.objects.filter(conversation=conversation)
            deleted_message_count = chat_messages.count()
            chat_messages.delete()
            
            # 대화기록 삭제
            conversation_title = conversation.title
            conversation.delete()
            
            print(f"DEBUG: 대화기록 삭제 완료 - ID: {conversation_id}, 제목: {conversation_title}, 삭제된 메시지 수: {deleted_message_count}")
            
            return Response({
                'success': True,
                'message': f'대화기록 "{conversation_title}"이(가) 삭제되었습니다.',
                'data': {
                    'deleted_conversation_id': str(conversation_id),
                    'deleted_message_count': deleted_message_count
                }
            }, status=status.HTTP_200_OK)
            
        except ValueError:
            return Response(
                {'success': False, 'message': '토큰 형식이 올바르지 않습니다.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            print(f"DEBUG: 대화기록 삭제 중 오류 발생: {str(e)}")
            return Response(
                {'success': False, 'message': f'대화기록 삭제 중 오류가 발생했습니다: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ChatQueryView(generics.CreateAPIView):
    """
    질문 전송 및 응답 생성
    """
    authentication_classes = []  # 인증 클래스 제외
    permission_classes = [AllowAny]  # 개발 단계에서는 인증 우회
    serializer_class = ChatQuerySerializer

    def create(self, request, *args, **kwargs):
        conversation_id = kwargs.get('conversation_id')
        
        # 디버깅을 위한 로그 추가
        print(f"DEBUG: ChatQueryView.create() 호출됨")
        print(f"DEBUG: conversation_id = {conversation_id}")
        print(f"DEBUG: kwargs = {kwargs}")
        print(f"DEBUG: request.path = {request.path}")
        
        # JWT 토큰에서 사용자 정보 추출
        auth_header = request.headers.get('Authorization')
        user_id = None
        
        if auth_header:
            try:
                token_type, token = auth_header.split(' ')
                if token_type.lower() == 'bearer':
                    payload = verify_token(token)
                    if payload:
                        user_id = payload.get('user_id')
            except:
                pass
        
        try:
            # 디버깅을 위한 로그 추가
            print(f"DEBUG: Conversation 조회 시도 - conversation_id: {conversation_id}")
            print(f"DEBUG: Conversation 조회 시도 - user_id: {user_id}")
            
            # 데이터베이스에 해당 ID가 존재하는지 확인
            all_conversations = Conversation.objects.all()
            print(f"DEBUG: 데이터베이스의 모든 Conversation: {list(all_conversations.values('id', 'user_id', 'title'))}")
            
            # 🔒 보안 강화: user_id와 conversation_id를 모두 확인
            try:
                if user_id:
                    # JWT 토큰이 있는 경우: user_id와 conversation_id 모두 확인
                    conversation = Conversation.objects.get(
                        id=conversation_id,
                        user_id=user_id  # 반드시 본인의 대화방만 접근 가능
                    )
                    print(f"DEBUG: Conversation 조회 성공 (user_id + conversation_id): {conversation}")
                else:
                    # JWT 토큰이 없는 경우: conversation_id만으로 조회 (개발 단계)
                    conversation = Conversation.objects.get(id=conversation_id)
                    print(f"DEBUG: Conversation 조회 성공 (conversation_id만): {conversation}")
                    print(f"⚠️ 경고: JWT 토큰이 없어 보안 검증을 건너뜁니다!")
            except Conversation.DoesNotExist:
                print(f"DEBUG: Conversation.DoesNotExist 예외 발생!")
                print(f"DEBUG: 조회하려던 conversation_id: {conversation_id}")
                print(f"DEBUG: 조회하려던 user_id: {user_id}")
                return Response({
                    'success': False,
                    'message': '대화방을 찾을 수 없습니다',
                    'errors': {'conversation_id': '유효하지 않은 대화방 ID입니다.'}
                }, status=status.HTTP_404_NOT_FOUND
                )
                
        except Exception as e:
            print(f"DEBUG: 예상치 못한 오류 발생: {str(e)}")
            return Response({
                'success': False,
                'message': '대화방 조회 중 오류가 발생했습니다',
                'errors': {'error': str(e)}
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user_message = serializer.validated_data['message']

        # 사용자 메시지 저장
        user_message_obj = ChatMessage.objects.create(
            conversation=conversation,
            sender_type='user',
            content=user_message
        )

        # 첫 질문인 경우 대화기록 제목을 질문 내용으로 설정
        if conversation.messages.count() == 1:  # 방금 생성된 사용자 메시지가 첫 번째 메시지
            # 질문 내용을 제목으로 사용 (최대 50자로 제한)
            title = user_message[:50] + "..." if len(user_message) > 50 else user_message
            conversation.title = title
            conversation.save()
            print(f"DEBUG: 첫 질문으로 대화기록 제목 설정: {title}")

        try:
            print(f"DEBUG: 향상된 RAG 시스템 시작 - 질문: {user_message}")
            
            # 대화 히스토리 조회 (최근 10개 메시지)
            recent_messages = conversation.messages.all().order_by('-created_at')[:10]
            conversation_history = []
            
            for msg in reversed(recent_messages):  # 시간순으로 정렬
                # [SYSTEM] 메시지는 system 역할로 변환
                if msg.content.startswith('[SYSTEM]'):
                    conversation_history.append({
                        'role': 'system',
                        'content': msg.content[8:]  # '[SYSTEM]' 제거
                    })
                else:
                    conversation_history.append({
                        'role': 'user' if msg.sender_type == 'user' else 'assistant',
                        'content': msg.content
                    })
            
            print(f"DEBUG: 대화 히스토리 ({len(conversation_history)}개 메시지): {conversation_history}")
            
            # 향상된 RAG 시스템을 통한 답변 생성 (대화 히스토리 포함)
            rag_result = rag_answer_enhanced(user_message, conversation_history=conversation_history)
            
            # 사용자 정보가 있으면 데이터베이스에 저장
            if rag_result.get('metadata', {}).get('user_info'):
                user_info = rag_result['metadata']['user_info']
                # 이미 저장된 사용자 정보가 있는지 확인
                existing_system_msg = conversation.messages.filter(
                    content__startswith='[SYSTEM] user_context:'
                ).first()
                
                if not existing_system_msg:
                    # 사용자 정보를 데이터베이스에 저장
                    import json
                    context_content = f"user_context: {json.dumps(user_info, ensure_ascii=False)}"
                    ChatMessage.objects.create(
                        conversation=conversation,
                        sender_type='ai',
                        content=f"[SYSTEM] {context_content}"
                    )
                    print(f"DEBUG: 사용자 정보를 데이터베이스에 저장 완료: {user_info}")
            
            if rag_result.get("rag_used", False):
                ai_response = rag_result["answer"]
                sources = rag_result["sources"]
                print(f"DEBUG: 향상된 RAG 시스템 완료 - 응답: {ai_response[:100]}...")
                print(f"DEBUG: 사용된 도메인: {rag_result.get('question_type', {}).get('type', 'unknown')}")
            else:
                # RAG를 사용하지 않은 경우 (예: 인사말)
                ai_response = rag_result["answer"]
                sources = []
                print(f"DEBUG: RAG 없이 응답 생성 - 응답: {ai_response[:100]}...")
                
        except Exception as e:
            print(f"DEBUG: 향상된 RAG 시스템 실패 - 오류: {str(e)}")
            # 향상된 RAG 시스템 실패 시 기존 RAG 시스템으로 fallback
            try:
                print(f"DEBUG: 기존 RAG 시스템으로 fallback 시도")
                rag = rag_answer(user_message)
                ai_response = rag["answer"]
                sources = rag["sources"]
                print(f"DEBUG: 기존 RAG 시스템 fallback 성공")
            except Exception as fallback_error:
                print(f"DEBUG: 기존 RAG 시스템도 실패 - 오류: {str(fallback_error)}")
                # 모든 RAG 시스템 실패 시 기본 AI 응답 생성
                ai_response = f"죄송합니다. AI 시스템이 일시적으로 응답할 수 없습니다. 잠시 후 다시 시도해 주세요."
                sources = []

        # AI 응답 저장
        ai_msg = ChatMessage.objects.create(
            conversation=conversation,
            sender_type='ai',
            content=ai_response
        )
        
        # 대화방 업데이트 시간 갱신
        conversation.save()

        return Response({
            "response": ai_response,
            "message_id": str(ai_msg.id),
            "sources": sources,
            "conversation_title": conversation.title,  # 업데이트된 제목 반환
        }, status=status.HTTP_200_OK)

class ChatStatusView(generics.RetrieveAPIView):
    """
    응답 처리 상태 확인
    """
    # permission_classes = [IsAuthenticated]
    permission_classes = [AllowAny]

    def retrieve(self, request, *args, **kwargs):
        session_id = kwargs.get('session_id')
        try:
            conversation = Conversation.objects.get(id=session_id, user=request.user)
            last_message = conversation.messages.last()
            return Response({
                "status": "completed",
                "last_message": last_message.content if last_message else None,
                "last_updated": conversation.updated_at
            })
        except Conversation.DoesNotExist:
            return Response(
                {"error": "대화방을 찾을 수 없습니다."},
                status=status.HTTP_404_NOT_FOUND
            )


class ChatHistoryView(generics.RetrieveAPIView):
    """
    대화 히스토리 조회
    """
    authentication_classes = []
    permission_classes = [AllowAny]
    serializer_class = ChatMessageSerializer

    def retrieve(self, request, *args, **kwargs):
        conversation_id = kwargs.get('conversation_id')
        
        # JWT 토큰에서 사용자 정보 추출
        auth_header = request.headers.get('Authorization')
        user_id = None
        
        if auth_header:
            try:
                token_type, token = auth_header.split(' ')
                if token_type.lower() == 'bearer':
                    payload = verify_token(token)
                    if payload:
                        user_id = payload.get('user_id')
            except:
                pass
        
        try:
            # 보안 검증: user_id와 conversation_id를 모두 확인
            if user_id:
                conversation = Conversation.objects.get(
                    id=conversation_id,
                    user_id=user_id
                )
            else:
                # 개발 단계에서는 conversation_id만으로 조회
                conversation = Conversation.objects.get(id=conversation_id)
            
            # 대화 히스토리 조회 (최근 20개 메시지)
            messages = conversation.messages.all().order_by('created_at')[:20]
            serializer = self.get_serializer(messages, many=True)
            
            return Response({
                'success': True,
                'conversation_id': str(conversation.id),
                'conversation_title': conversation.title,
                'messages': serializer.data,
                'total_count': conversation.messages.count()
            })
            
        except Conversation.DoesNotExist:
            return Response({
                'success': False,
                'message': '대화방을 찾을 수 없습니다'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'success': False,
                'message': f'대화 히스토리 조회 중 오류가 발생했습니다: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatReportView(generics.CreateAPIView):
    serializer_class = ChatReportSerializer
    authentication_classes = []  # 커스텀 JWT 인증을 사용하므로 DRF 인증 비활성화
    permission_classes = [AllowAny]  # 커스텀 인증을 사용하므로 AllowAny

    @require_auth
    def create(self, request, *args, **kwargs):
        chat_id = kwargs.get("chat_id")
        logger.info(f"chat_id: kwargs.get 실행 결과: {chat_id}")
        if not chat_id:
            return Response({"error": "chat_id 필요"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # 신고 대상 메시지 가져오기
            message = ChatMessage.objects.get(id=chat_id)
            print(f"message from DB: {message}, id: {message.id}, type: {type(message.id)}")
        except ChatMessage.DoesNotExist:
            return Response({"error": "메시지 없음"}, status=status.HTTP_404_NOT_FOUND)
        except ValueError:
            return Response({"error": "chat_id 형식 오류"}, status=status.HTTP_400_BAD_REQUEST)

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # create 호출 시 chat 객체를 kwargs로 넘김
        serializer.save(chat=message)

        return Response(serializer.data, status=status.HTTP_201_CREATED)

class FormDownloadView(generics.GenericAPIView):
    """
    서식 파일 다운로드 API
    """
    authentication_classes = []  # 인증 클래스 제외
    permission_classes = [AllowAny]  # 개발 단계에서는 인증 우회
    
    def get(self, request, *args, **kwargs):
        """
        S3에서 서식 파일을 다운로드하여 반환
        """
        s3_key = request.query_params.get('s3_key')
        
        if not s3_key:
            return Response({
                'success': False,
                'message': 'S3 키가 제공되지 않았습니다.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # S3 클라이언트 초기화
            s3_client = boto3.client(
                's3',
                region_name=os.getenv('AWS_REGION', 'ap-northeast-2'),
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
            
            bucket_name = os.getenv('AWS_S3_BUCKET_NAME', 'companypolicy')
            
            # S3에서 파일 다운로드
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            file_content = response['Body'].read()
            
            # 파일명 추출 (S3 키에서 마지막 부분)
            filename = s3_key.split('/')[-1]
            
            # 파일 확장자에 따른 Content-Type 설정
            if filename.lower().endswith('.pdf'):
                content_type = 'application/pdf'
            elif filename.lower().endswith('.doc'):
                content_type = 'application/msword'
            elif filename.lower().endswith('.docx'):
                content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            elif filename.lower().endswith('.xls'):
                content_type = 'application/vnd.ms-excel'
            elif filename.lower().endswith('.xlsx'):
                content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            else:
                content_type = 'application/octet-stream'
            
            # HTTP 응답 생성
            http_response = HttpResponse(file_content, content_type=content_type)
            http_response['Content-Disposition'] = f'attachment; filename="{filename}"'
            http_response['Content-Length'] = len(file_content)
            
            return http_response
            
        except ClientError as e:
            logger.error(f"S3 파일 다운로드 실패: {e}")
            return Response({
                'success': False,
                'message': '파일을 찾을 수 없거나 다운로드할 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)
            
        except Exception as e:
            logger.error(f"파일 다운로드 중 오류: {e}")
            return Response({
                'success': False,
                'message': '파일 다운로드 중 오류가 발생했습니다.'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

