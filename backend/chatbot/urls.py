# chatbot/urls.py
from django.urls import path
from .views import (
    ConversationListView,
    ConversationCreateView,
    ConversationDeleteView,
    ChatQueryView,
    ChatHistoryView,
    ChatStatusView,
    ChatReportView,
    FormDownloadView
)

urlpatterns = [
    path('list/', ConversationListView.as_view(), name='conversation-list'),
    path('new/', ConversationCreateView.as_view(), name='conversation-create'),
    path('<uuid:conversation_id>/delete/', ConversationDeleteView.as_view(), name='conversation-delete'),
    path('<uuid:conversation_id>/query/', ChatQueryView.as_view(), name='chat-query'),
    path('<uuid:conversation_id>/history/', ChatHistoryView.as_view(), name='chat-history'),
    path('<uuid:conversation_id>/status/', ChatStatusView.as_view(), name='chat-status'),
    path('<uuid:chat_id>/report/', ChatReportView.as_view(), name='chat-report'),
    path('form/download/', FormDownloadView.as_view(), name='form-download'),
]