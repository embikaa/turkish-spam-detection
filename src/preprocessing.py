import re
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('turkish'))


def clean_text(text: str) -> str:
    
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    text = re.sub(r'[^\w\s]', ' ', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_text_for_bert(text: str) -> str:
   
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_text_for_tfidf(text: str) -> str:
    
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    text = re.sub(r'[^\w\s]', ' ', text)
    
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    try:
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [w for w in tokens if w not in STOP_WORDS]
        return " ".join(filtered_tokens)
    except Exception:
        return text