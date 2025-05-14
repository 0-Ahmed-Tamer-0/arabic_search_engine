# from camel_tools.utils import normalize
# from camel_tools.tokenizers.word import simple_word_tokenize
# from camel_tools import STOPWORDS
# from transformers import AutoTokenizer, AutoModel
# import torch

# class ArabicProcessor:
#     def __init__(self):
#         # Initialize Arabic NLP tools
#         self.tokenizer = simple_word_tokenize()
#         self.stopwords = set(STOPWORDS['ar'])
#         self.bert_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
#         self.bert_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv02")

#     def preprocess(self, text):
#         """Clean and normalize Arabic text"""
#         # Normalize characters
#         text = normalize(text)
#         text = text.replace('آ', 'ا').replace('أ', 'ا').replace('إ', 'ا')
        
#         # Tokenize and remove stopwords
#         tokens = self.tokenizer.tokenize(text)
#         tokens = [t for t in tokens if t not in self.stopwords]
        
#         return ' '.join(tokens)
    
#     def get_embedding(self, text):
#         """Generate BERT embedding for text"""
#         processed_text = self.preprocess(text)
#         inputs = self.bert_tokenizer(
#             processed_text,
#             return_tensors="pt",
#             max_length=512,
#             truncation=True,
#             padding='max_length'
#         )
#         with torch.no_grad():
#             outputs = self.bert_model(**inputs)
#         return outputs.last_hidden_state[:,0,:].numpy()
#----------working code-------------
# utils.py
import re
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
import torch

# Download NLTK Arabic stopwords data
# nltk.download('stopwords')

class ArabicProcessor:
    def __init__(self):
        # Initialize Arabic stopwords from NLTK
        self.stopwords = set(stopwords.words('arabic'))
        
        # Initialize AraBERT components
        self.bert_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
        self.bert_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv02")
        
        # Arabic character normalization mappings
        self.normalization_map = {
            'أ': 'ا',
            'إ': 'ا',
            'آ': 'ا',
            'ة': 'ه',
            'ى': 'ي',
            'ئ': 'ء',
            'ؤ': 'ء'
        }
        self.special_cases = {
            'القرآن': 'القرآن',  # Prevent normalization
            'الله': 'الله'
        }

    def normalize(self, text):
        """Normalize Arabic text by:
        1. Replacing character variants with base forms
        2. Removing diacritics (harakat)
        3. Trimming whitespace
        """
         # Handle special cases first
        for word, replacement in self.special_cases.items():
            text = text.replace(word, replacement)
            
        # Character normalization
        for char, replacement in self.normalization_map.items():
            text = text.replace(char, replacement)
        
        # Remove diacritics using regex (harakat, sukun, etc.)
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
        
        return text.strip()

    def tokenize(self, text):
        """Basic Arabic word tokenizer using regex"""
        return re.findall(r"[\w']+", text)

    def remove_stopwords(self, tokens):
        """Filter out Arabic stopwords"""
        return [token for token in tokens if token not in self.stopwords]

    def preprocess(self, text):
        """Full text processing pipeline"""
        # Normalization
        normalized_text = self.normalize(text)
        
        # Tokenization
        tokens = self.tokenize(normalized_text)
        
        # Stopword removal
        filtered_tokens = self.remove_stopwords(tokens)
        
        return ' '.join(filtered_tokens)

    def get_embedding(self, text):
        """Generate BERT embedding for processed text"""
        processed_text = self.preprocess(text)
        
        inputs = self.bert_tokenizer(
            processed_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding='max_length'
        )
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            
        # Use [CLS] token embedding as document representation
        return outputs.last_hidden_state[:,0,:].numpy()
#====end of working code---------------------------

# utils.py
# import re
# import nltk
# from nltk.corpus import stopwords
# from camel_tools.disambig.mle import MLEDisambiguator
# from camel_tools.tokenizers.word import simple_word_tokenize
# from transformers import AutoTokenizer, AutoModel
# import torch

# # nltk.download('stopwords')

# class ArabicProcessor:
#     def __init__(self):
#         self.stopwords = set(stopwords.words('arabic'))
#         self.mle = MLEDisambiguator.pretrained()
#         self.tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
#         self.model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv02")
        
#         # Normalization mappings
#         self.norm_map = {
#             'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ة': 'ه', 
#             'ى': 'ي', 'ئ': 'ء', 'ؤ': 'ء'
#         }

#     def normalize(self, text):
#         """Arabic text normalization"""
#         for char, repl in self.norm_map.items():
#             text = text.replace(char, repl)
#         text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
#         return text.strip()

#     def lemmatize(self, tokens):
#         """Arabic lemmatization using camel-tools"""
#         disambig = self.mle.disambiguate(tokens)
#         return [d.analyses[0].analysis['lex'] if d.analyses else t 
#                 for d, t in zip(disambig, tokens)]

#     def preprocess(self, text):
#         """Full preprocessing pipeline"""
#         text = self.normalize(text)
#         tokens = simple_word_tokenize(text)
#         tokens = [t for t in tokens if t not in self.stopwords]
#         return ' '.join(self.lemmatize(tokens))

#     def get_embedding(self, text):
#         """Generate BERT embedding for processed text"""
#         processed = self.preprocess(text)
#         inputs = self.tokenizer(
#             processed,
#             return_tensors="pt",
#             max_length=512,
#             truncation=True,
#             padding='max_length'
#         )
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         return outputs.last_hidden_state[:,0,:].numpy()

# class HybridRetriever:
#     def __init__(self, documents):
#         self.inverted_index = self.build_inverted_index(documents)
#         self.doc_embeddings = []
        
#     def build_inverted_index(self, documents):
#         """Create inverted index from PDF example (modified for Arabic)"""
#         index = {}
#         for doc_id, text in documents.items():
#             tokens = simple_word_tokenize(text)
#             clean_tokens = [t for t in tokens if t not in set(stopwords.words('arabic'))]
#             for token in clean_tokens:
#                 if token not in index:
#                     index[token] = set()
#                 index[token].add(doc_id)
#         return index

    # def bert_retrieval(self, query, top_k=10):
    #     """BERT-based semantic search"""
    #     query_embedding = self.get_embedding(query)
    #     # FAISS search implementation here
    #     return sorted_results

    # def keyword_retrieval(self, query):
    #     """Traditional keyword search"""
    #     query_terms = self.preprocess(query).split()
    #     results = set()
    #     for term in query_terms:
    #         if term in self.inverted_index:
    #             results.update(self.inverted_index[term])
    #     return list(results)

    # def hybrid_search(self, query):
    #     """Combine BERT and keyword results"""
    #     semantic_results = self.bert_retrieval(query)
    #     keyword_results = self.keyword_retrieval(query)
    #     # Combine and re-rank results
    #     return final_results