import re
import nltk
from nltk.corpus import stopwords

# Gerekli NLTK dosyalarını indir
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
    
    # Emojileri ve özel karakterleri temizle
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text) # Rakamları temizle (isteğe bağlı)
    text = re.sub(r'\s+', ' ', text).strip()

    # Stopwords temizliği (isteğe bağlı, bazen bağlamı bozar ama genelde iyidir)
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [w for w in tokens if w not in STOP_WORDS]
    
    return " ".join(filtered_tokens)