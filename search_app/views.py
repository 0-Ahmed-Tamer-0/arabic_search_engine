from django.shortcuts import render
from .models import Document, QueryLog
from .utils import ArabicProcessor
import faiss
import numpy as np

processor = ArabicProcessor()
faiss_index = None
doc_ids = []

def initialize_index():
    """Initialize or rebuild the FAISS index"""
    global faiss_index, doc_ids
    try:
        documents = Document.objects.all().order_by('id')
        if not documents.exists():
            return
        
        embeddings = [doc.get_embedding() for doc in documents]
        embeddings = np.vstack(embeddings).astype('float32')
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        faiss_index = index
        doc_ids = [doc.id for doc in documents]
        
    except Exception as e:
        print(f"Error initializing index: {e}")
        faiss_index = None
        doc_ids = []

# Initialize index on server start
initialize_index()

def search_view(request):
    global faiss_index, doc_ids
    
    if request.method == 'GET':
        query = request.GET.get('q', '').strip()
        
        # Rebuild index if not initialized
        if faiss_index is None:
            initialize_index()
            
        if query:
            if faiss_index is None:
                return render(request, 'search/results.html', {
                    'error': 'Search index not ready. Please add documents first.',
                    'query': query
                })
            
            try:
                query_embedding = processor.get_embedding(query)
                scores, indices = faiss_index.search(query_embedding, 10)
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    try:
                        doc = Document.objects.get(id=doc_ids[idx])
                        results.append({
                            'text': doc.text,
                            'score': float(score)
                        })
                    except (Document.DoesNotExist, IndexError):
                        continue
                
                QueryLog.objects.create(query=query, results_count=len(results))
                
                return render(request, 'search/results.html', {
                    'results': results,
                    'query': query
                })
                
            except Exception as e:
                print(f"Search error: {e}")
                return render(request, 'search/results.html', {
                    'error': 'An error occurred during search.',
                    'query': query
                })
    
    return render(request, 'search/search.html')
    
    
# def search_view(request):
#     if request.method == 'GET':
#         query = request.GET.get('q', '')
        
#         if query:
#             # Process query
#             query_embedding = processor.get_embedding(query)
            
#             # Search FAISS index
#             scores, indices = faiss_index.search(query_embedding, 10)
            
#             # Get documents by ID
#             results = []
#             for score, idx in zip(scores[0], indices[0]):
#                 try:
#                     doc = Document.objects.get(id=doc_ids[idx])
#                     results.append({
#                         'text': doc.text,
#                         'score': float(score),
#                         'original': doc.text  # Show original text
#                     })
#                 except Document.DoesNotExist:
#                     continue
            
#             # Log query
#             QueryLog.objects.create(
#                 query=query,
#                 results_count=len(results)
#             )
            
#             return render(request, 'search/results.html', {
#                 'results': results,
#                 'query': query
#             })
        
#     return render(request, 'search/search.html')
# views.py
# from django.shortcuts import render
# from .models import Document
# from .utils import ArabicProcessor, HybridRetriever

# processor = ArabicProcessor()
# documents = {doc.id: doc.text for doc in Document.objects.all()}
# retriever = HybridRetriever(documents)

# def search_view(request):
#     if request.method == 'GET':
#         query = request.GET.get('q', '')
#         if query:
#             results = retriever.keyword_retrieval(query)
#             return render(request, 'results.html', {'results': results})
#     return render(request, 'search.html')