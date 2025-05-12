from django.db import models
import numpy as np
import pickle

class Document(models.Model):
    text = models.TextField(verbose_name="النص الكامل")
    embedding = models.BinaryField(verbose_name="التضمين النصي")
    created_at = models.DateTimeField(auto_now_add=True)
    
    def set_embedding(self, vector):
        """Serialize numpy array to binary"""
        self.embedding = pickle.dumps(vector)
    
    def get_embedding(self):
        """Deserialize binary to numpy array"""
        return pickle.loads(self.embedding)
    
    def save(self, *args, **kwargs):
        # Auto-generate embedding on save
        if not self.embedding:
            from .utils import ArabicProcessor
            processor = ArabicProcessor()
            embedding = processor.get_embedding(self.text)
            self.set_embedding(embedding)
        super().save(*args, **kwargs)
    
    class Meta:
        verbose_name = "وثيقة"
        verbose_name_plural = "الوثائق"

class QueryLog(models.Model):
    query = models.CharField(max_length=255)
    results_count = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "سجل البحث"
        verbose_name_plural = "سجلات البحث"