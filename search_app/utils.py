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