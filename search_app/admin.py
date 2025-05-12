from django.contrib import admin
from .models import Document, QueryLog

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('id', 'created_at')
    search_fields = ('text',)

@admin.register(QueryLog)
class QueryLogAdmin(admin.ModelAdmin):
    list_display = ('query', 'results_count', 'created_at')
    list_filter = ('created_at',)