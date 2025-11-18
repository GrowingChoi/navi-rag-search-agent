from rest_framework import serializers
from .models import Conversation, ChatMessage, ChatReport
from authapp.models import UserInfo

class ChatMessageSerializer(serializers.ModelSerializer):
    sender_type_display = serializers.CharField(
        source='get_sender_type_display', 
        read_only=True
    )
    
    class Meta:
        model = ChatMessage
        fields = ['id', 'sender_type', 'sender_type_display', 'content', 'created_at']
        read_only_fields = ['id', 'created_at']

class ChatQuerySerializer(serializers.Serializer):
    message = serializers.CharField(
        max_length=1000,
        error_messages={
            'required': '메시지를 입력해주세요.',
            'blank': '메시지를 입력해주세요.',
            'max_length': '메시지는 1000자 이내로 입력해주세요.'
        }
    )

    def validate_message(self, value):
        if not value.strip():
            raise serializers.ValidationError("메시지 내용을 입력해주세요.")
        return value

class ConversationSerializer(serializers.ModelSerializer):
    # messages 필드를 중첩된 관계로 추가
    messages = ChatMessageSerializer(many=True, read_only=True)

    class Meta:
        model = Conversation
        fields = ['id', 'title', 'created_at', 'updated_at', 'messages']
        read_only_fields = ['id', 'created_at', 'updated_at', 'messages']

    def validate_title(self, value):
        """제목 유효성 검사"""
        if len(value) < 2:
            raise serializers.ValidationError("제목은 2자 이상 입력해주세요.")
        return value

class ChatReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatReport
        fields = [
            'report_id',
            'chat',
            'error_type',
            'reason',
            'reported_by',
            'created_at',
            'solved_yn',
            'remark'
        ]
        read_only_fields = [
            'report_id',
            'reported_by',
            'created_at',
            'solved_yn',
            'remark',
            'chat'
        ]

    def create(self, validated_data):
        import logging
        import traceback
        logger = logging.getLogger(__name__)
        
        # 스택 트레이스 정보 추가
        stack_trace = traceback.format_stack()
        caller_info = stack_trace[-3] if len(stack_trace) > 3 else "Unknown"
        
        logger.info(f"🔍 ChatReport 생성 시작 - chat_id: {validated_data.get('chat')}")
        logger.info(f"📞 호출자 정보: {caller_info.strip()}")
        logger.info(f"📋 요청 데이터: {validated_data}")
        
        # UserInfo 기반 reported_by 설정
        request = self.context.get('request')
        
        # 커스텀 인증 데코레이터에서 설정된 사용자 정보 사용
        if hasattr(request, 'user_data') and request.user_data:
            user_uuid = request.user_data[0]  # user_id는 첫 번째 컬럼
        else:
            raise serializers.ValidationError("사용자 정보 없음")

        try:
            user_info = UserInfo.objects.get(user_id=user_uuid)
            validated_data['reported_by'] = user_info
            logger.info(f"👤 사용자 정보 설정 - user_id: {user_uuid}")
        except UserInfo.DoesNotExist:
            raise serializers.ValidationError("사용자를 찾을 수 없음")

        # chat 확인
        chat = validated_data.get('chat')
        if chat:
            logger.info(f"✅ chat.id: {chat.id}")

        # 신고 사유가 비어있으면 기본값 설정
        if not validated_data.get('reason') or not validated_data.get('reason').strip():
            validated_data['reason'] = f"사용자가 {validated_data.get('error_type', 'unknown')} 유형으로 신고함"
            logger.info(f"📝 기본 신고 사유 설정: {validated_data['reason']}")

        # ChatReport 저장
        logger.info(f"🔧 super().create(validated_data) 호출 시작")
        logger.info(f"📋 저장할 데이터: {validated_data}")
        chat_report = super().create(validated_data)
        logger.info(f"💾 ChatReport 생성 완료 - report_id: {chat_report.report_id}")
        logger.info(f"📅 생성 시간: {chat_report.created_at}")
        logger.info(f"👤 신고자: {chat_report.reported_by}")
        logger.info(f"💬 신고 사유: {chat_report.reason}")
        logger.info(f"🏷️ 신고 유형: {chat_report.error_type}")

        # ChatMessage.report = 'Y' 업데이트
        if chat:
            logger.info(f"📝 chat.report = 'Y' 업데이트 시작")
            chat.report = "Y"
            chat.save(update_fields=['report'])
            logger.info(f"✅ chat.report 업데이트 완료")

        return chat_report


