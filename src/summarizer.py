"""Extractive Text Summarization for T20 Cricket News.

Implements extractive summarization using sentence-level word-frequency scoring,
cricket entity weighting (scores, wickets, overs), position bias, and sentence
deduplication.
"""

from typing import Dict, List, Set, Tuple
import re

from src.preprocessing import clean_text, get_stopwords, tokenize_sentences, tokenize_words


def jaccard_similarity(sent1_words: Set[str], sent2_words: Set[str]) -> float:
    """Compute Jaccard similarity between two word sets for deduplication."""
    if not sent1_words or not sent2_words:
        return 0.0
    intersection = len(sent1_words.intersection(sent2_words))
    union = len(sent1_words.union(sent2_words))
    return intersection / union if union > 0 else 0.0


def calculate_sentence_score(
    sentence: str,
    index: int,
    total_sentences: int,
    word_frequencies: Dict[str, float],
    stopwords: Set[str],
) -> float:
    """Score a single sentence based on term frequencies, position bias, and cricket entities."""
    words = tokenize_words(sentence, remove_stops=False, lowercase=True)
    if not words or len(words) < 4:
        return 0.0

    content_words = [w for w in words if w not in stopwords and len(w) > 2]
    if not content_words:
        return 0.0

    # Base score: average term frequency of content words
    base_score = sum(word_frequencies.get(w, 0.0) for w in content_words) / len(content_words)

    # Lead position bias: Sports journalism places key match results in lead sentences
    position_factor = 1.0 / (1.0 + 0.15 * index)

    # Cricket entity boost: Give slight prominence to scores, wickets, runs, and overs
    has_cricket_stats = bool(
        re.search(r"\b(\d+/\d+|\d+-\w+|\d+\s*runs|\d+\s*wickets|\d+\s*overs|super over|final|champions)\b", sentence.lower())
    )
    stat_boost = 1.25 if has_cricket_stats else 1.0

    return float(base_score * position_factor * stat_boost)


def summarize_text(
    text: str,
    max_sentences: int = 3,
    min_sentence_length: int = 4,
) -> Dict[str, object]:
    """Generate an extractive summary of cricket news text.

    Args:
        text: Raw or cleaned article text.
        max_sentences: Target sentence count for summary (typically 2 to 5).
        min_sentence_length: Minimum word count for a sentence to be eligible.

    Returns:
        Dict containing:
            - summary: The formatted summary string
            - original_word_count: Total words in input
            - summary_word_count: Total words in summary
            - reduction_percentage: Percentage reduction in word length
            - original_sentences: Count of original sentences
            - summary_sentences: Count of selected sentences
    """
    cleaned = clean_text(text, preserve_case=True)
    if not cleaned:
        return {
            "summary": "No article content provided to summarize.",
            "original_word_count": 0,
            "summary_word_count": 0,
            "reduction_percentage": 0.0,
            "original_sentences": 0,
            "summary_sentences": 0,
        }

    sentences = tokenize_sentences(cleaned)
    original_word_count = len(cleaned.split())

    # Filter out very short fragments
    valid_sentences = [s for s in sentences if len(s.split()) >= min_sentence_length]

    # Handle short articles gracefully
    if len(valid_sentences) <= max_sentences:
        summary_str = " ".join(valid_sentences) if valid_sentences else cleaned
        summary_word_count = len(summary_str.split())
        reduction = (
            round((1.0 - (summary_word_count / max(original_word_count, 1))) * 100, 1)
            if original_word_count > summary_word_count
            else 0.0
        )
        return {
            "summary": summary_str,
            "original_word_count": original_word_count,
            "summary_word_count": summary_word_count,
            "reduction_percentage": max(0.0, reduction),
            "original_sentences": len(sentences),
            "summary_sentences": len(valid_sentences),
        }

    stopwords = get_stopwords()
    all_words = tokenize_words(cleaned, remove_stops=True, lowercase=True)

    # Compute raw frequency dictionary
    raw_frequencies: Dict[str, int] = {}
    for word in all_words:
        raw_frequencies[word] = raw_frequencies.get(word, 0) + 1

    max_freq = max(raw_frequencies.values()) if raw_frequencies else 1
    normalized_frequencies = {w: count / max_freq for w, count in raw_frequencies.items()}

    # Score each sentence
    scored_sentences: List[Tuple[float, int, str]] = []
    sentence_word_sets: List[Set[str]] = []

    for idx, sentence in enumerate(valid_sentences):
        score = calculate_sentence_score(
            sentence=sentence,
            index=idx,
            total_sentences=len(valid_sentences),
            word_frequencies=normalized_frequencies,
            stopwords=stopwords,
        )
        scored_sentences.append((score, idx, sentence))
        sentence_word_sets.append(set(tokenize_words(sentence, remove_stops=True, lowercase=True)))

    # Sort descending by score
    sorted_by_score = sorted(scored_sentences, key=lambda x: x[0], reverse=True)

    # Select top sentences while avoiding duplicate information (Jaccard overlap > 0.65)
    selected: List[Tuple[float, int, str]] = []
    selected_word_sets: List[Set[str]] = []

    for candidate in sorted_by_score:
        if len(selected) >= max_sentences:
            break

        cand_idx = candidate[1]
        cand_words = sentence_word_sets[cand_idx]

        # Check redundancy against already chosen sentences
        is_redundant = False
        for chosen_words in selected_word_sets:
            if jaccard_similarity(cand_words, chosen_words) > 0.65:
                is_redundant = True
                break

        if not is_redundant:
            selected.append(candidate)
            selected_word_sets.append(cand_words)

    # If deduplication dropped too many, fill up to max_sentences
    if len(selected) < max_sentences:
        for candidate in sorted_by_score:
            if len(selected) >= max_sentences:
                break
            if candidate not in selected:
                selected.append(candidate)

    # Restore natural chronological order
    selected_in_order = sorted(selected, key=lambda x: x[1])
    summary_text = " ".join(item[2] for item in selected_in_order)
    summary_word_count = len(summary_text.split())

    reduction = (
        round((1.0 - (summary_word_count / max(original_word_count, 1))) * 100, 1)
        if original_word_count > summary_word_count
        else 0.0
    )

    return {
        "summary": summary_text,
        "original_word_count": original_word_count,
        "summary_word_count": summary_word_count,
        "reduction_percentage": max(0.0, reduction),
        "original_sentences": len(sentences),
        "summary_sentences": len(selected_in_order),
    }
