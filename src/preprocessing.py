"""Text preprocessing pipeline for T20 Cricket World Cup articles.

Provides text cleaning, HTML stripping, sentence tokenization, word tokenization,
and stopword removal with graceful fallbacks and cricket terminology preservation.
"""

import html
import re
from typing import List, Set

# Attempt safe NLTK resource initialization with regex fallbacks
NLTK_AVAILABLE = False
_NLTK_STOPWORDS: Set[str] = set()

# Built-in fallback stopwords if NLTK is unavailable or downloads are restricted
BUILTIN_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize

    # Download required datasets quietly if missing
    for resource in ["punkt", "punkt_tab", "stopwords"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

    try:
        _NLTK_STOPWORDS = set(stopwords.words("english"))
        NLTK_AVAILABLE = True
    except Exception:
        _NLTK_STOPWORDS = BUILTIN_STOPWORDS
        NLTK_AVAILABLE = False
except ImportError:
    NLTK_AVAILABLE = False
    _NLTK_STOPWORDS = BUILTIN_STOPWORDS


def get_stopwords() -> Set[str]:
    """Return effective stopword set (NLTK if loaded, else robust built-in)."""
    return _NLTK_STOPWORDS if _NLTK_STOPWORDS else BUILTIN_STOPWORDS


def clean_text(text: str, preserve_case: bool = True) -> str:
    """Clean raw article text: strip HTML, unescape entities, remove URLs, and normalize whitespace.

    Preserves important cricket expressions such as scores (176/7), bowling figures
    (4-0-18-2), batting knocks (76 off 59 balls), and proper player/venue names.
    """
    if not text or not isinstance(text, str):
        return ""

    # Unescape HTML entities (e.g., &amp; -> &)
    cleaned = html.unescape(text)

    # Remove HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    # Remove URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # Normalize typographic quotes and dashes
    cleaned = cleaned.replace("“", '"').replace("”", '"')
    cleaned = cleaned.replace("‘", "'").replace("’", "'")
    cleaned = cleaned.replace("—", " - ").replace("–", " - ")

    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not preserve_case:
        cleaned = cleaned.lower()

    return cleaned


def tokenize_sentences(text: str) -> List[str]:
    """Tokenize text into distinct sentences with fallback support."""
    cleaned = clean_text(text, preserve_case=True)
    if not cleaned:
        return []

    if NLTK_AVAILABLE:
        try:
            raw_sents = sent_tokenize(cleaned)
            sents = [s.strip() for s in raw_sents if len(s.strip()) > 0]
            if sents:
                return sents
        except Exception:
            pass

    # High-reliability regex fallback sentence tokenizer:
    # Splits after period, exclamation, or question mark followed by whitespace and capital/quote/digit
    raw_sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", cleaned)
    return [s.strip() for s in raw_sents if len(s.strip()) > 0]


def tokenize_words(text: str, remove_stops: bool = True, lowercase: bool = True) -> List[str]:
    """Tokenize words while retaining alphanumeric terms, scores, and cricket numbers."""
    cleaned = clean_text(text, preserve_case=not lowercase)
    if not cleaned:
        return []

    # Match tokens: alphanumeric words, hyphenated names, and scores (e.g. 176/7, 4-29)
    tokens = re.findall(r"\b[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*\b", cleaned)

    if lowercase:
        tokens = [t.lower() for t in tokens]

    if remove_stops:
        stops = get_stopwords()
        tokens = [t for t in tokens if t.lower() not in stops and len(t) > 1]

    return tokens


def preprocess_for_vectorizer(text: str) -> str:
    """Preprocess text for TF-IDF feature extraction."""
    words = tokenize_words(text, remove_stops=True, lowercase=True)
    return " ".join(words)


def is_valid_article(text: str, min_words: int = 10) -> bool:
    """Check if article text is substantial enough for summarization and clustering."""
    if not text or not isinstance(text, str):
        return False
    words = text.split()
    return len(words) >= min_words
