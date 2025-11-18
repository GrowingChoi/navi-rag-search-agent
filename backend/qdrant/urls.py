from django.urls import path
from . import views

app_name = 'qdrant'

urlpatterns = [
    path('add-document/', views.add_document, name='add_document'),
    path('search/', views.search_documents, name='search_documents'),
    path('collection-info/', views.collection_info, name='collection_info'),
    path('add-all-documents/', views.add_all_documents, name='add_all_documents'),
    path('categories/', views.get_categories, name='get_categories'),
    path('delete-collection/', views.delete_collection, name='delete_collection'),
]
