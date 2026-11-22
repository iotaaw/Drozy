"""
Text Preprocessing Module for Driver Fatigue Text Analysis.

Implements standard NLP preprocessing stages:
1. Lowercasing & Unicode Normalization
2. Punctuation & Symbol Cleaning
3. Word Tokenization
4. Stopword Removal
5. Lemmatization (with fallback to morphological stemmer)
"""

import re
import string

# Standard English stopwords (NLTK standard 179-word stopword list)
ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
    'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
    'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
    'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
    'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma',
    'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
}

# Try importing NLTK lemmatizer; if data is missing, fallback to PorterStemmer/Rule-based
_LEMMATIZER = None
_STEMMER = None

try:
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    # Test if WordNet corpus is loaded
    wordnet.synsets('drive')
    _LEMMATIZER = WordNetLemmatizer()
except Exception:
    _LEMMATIZER = None

try:
    from nltk.stem import PorterStemmer
    _STEMMER = PorterStemmer()
except Exception:
    _STEMMER = None

# Common driving & fatigue domain lemmatization mapping for clean canonical forms
DOMAIN_LEMMA_MAP = {
    'drowsy': 'drowsy',
    'drowsiness': 'drowsy',
    'sleepy': 'sleepy',
    'sleeping': 'sleep',
    'slept': 'sleep',
    'sleepiness': 'sleepy',
    'dozing': 'doze',
    'dozed': 'doze',
    'yawning': 'yawn',
    'yawns': 'yawn',
    'yawned': 'yawn',
    'eyelids': 'eyelid',
    'eyes': 'eye',
    'exhausted': 'exhaust',
    'exhaustion': 'exhaust',
    'fatigued': 'fatigue',
    'fatiguing': 'fatigue',
    'tiring': 'tire',
    'tired': 'tire',
    'tiredness': 'tire',
    'driving': 'drive',
    'drives': 'drive',
    'drove': 'drive',
    'driven': 'drive',
    'stressing': 'stress',
    'stressed': 'stress',
    'anxious': 'anxiety',
    'frustrated': 'frustrate',
    'headaches': 'headache',
    'stopping': 'stop',
    'stopped': 'stop',
    'resting': 'rest',
    'rested': 'rest',
    'breaks': 'break',
    'energetic': 'energy',
    'energized': 'energy',
    'refreshed': 'refresh',
    'refreshing': 'refresh',
    'attentive': 'attention',
}


def lemmatize_token(token: str) -> str:
    """Lemmatize a single token using WordNet or morphological fallbacks."""
    token = token.lower().strip()
    if token in DOMAIN_LEMMA_MAP:
        return DOMAIN_LEMMA_MAP[token]
    
    if _LEMMATIZER is not None:
        try:
            # Try verb first, then noun, then adjective
            lemma = _LEMMATIZER.lemmatize(token, pos='v')
            if lemma == token:
                lemma = _LEMMATIZER.lemmatize(token, pos='n')
            if lemma == token:
                lemma = _LEMMATIZER.lemmatize(token, pos='a')
            return lemma
        except Exception:
            pass

    # Suffix-based morphological normalization fallback
    if token.endswith('ing') and len(token) > 5:
        return token[:-3]
    if token.endswith('ed') and len(token) > 4:
        return token[:-2]
    if token.endswith('ly') and len(token) > 4:
        return token[:-2]
    if token.endswith('es') and len(token) > 4:
        return token[:-2]
    if token.endswith('s') and not token.endswith('ss') and len(token) > 3:
        return token[:-1]

    return token


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into alphanumeric word tokens."""
    return re.findall(r'\b[a-zA-Z]{2,}\b', text)


def preprocess_text(text: str) -> str:
    """
    End-to-end preprocessing pipeline for driver fatigue classification.
    Returns space-separated processed tokens ready for TF-IDF vectorization.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    lowered = text.lower()

    # 2. Tokenize (strips punctuation and digits)
    tokens = tokenize_text(lowered)

    # 3. Stopword removal & Lemmatization
    processed_tokens = []
    for tok in tokens:
        if tok not in ENGLISH_STOPWORDS:
            lemma = lemmatize_token(tok)
            if len(lemma) >= 2:
                processed_tokens.append(lemma)

    return " ".join(processed_tokens)


def demonstrate_preprocessing(text: str) -> dict:
    """
    Step-by-step preprocessing breakdown for educational UI and documentation.
    """
    raw = str(text) if text is not None else ""
    lowered = raw.lower()
    cleaned = re.sub(r'[^a-zA-Z\s]', ' ', lowered)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    tokens = tokenize_text(lowered)
    stopwords_removed = [tok for tok in tokens if tok not in ENGLISH_STOPWORDS]
    lemmatized = [lemmatize_token(tok) for tok in stopwords_removed]
    final_text = " ".join([w for w in lemmatized if len(w) >= 2])

    return {
        "raw_text": raw,
        "lowercased": lowered,
        "cleaned": cleaned,
        "tokens": tokens,
        "stopwords_removed": stopwords_removed,
        "lemmatized_tokens": lemmatized,
        "final_processed_text": final_text
    }
