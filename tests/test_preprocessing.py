"""Unit tests for text preprocessing."""

import unittest
from src.preprocessing import (
    clean_text,
    tokenize_sentences,
    tokenize_words,
    preprocess_for_vectorizer,
    is_valid_article,
)


class TestPreprocessing(unittest.TestCase):
    def test_clean_text_html_and_urls(self):
        raw = "<p>India defeated South Africa. Check out https://espncricinfo.com/t20 for more!</p>"
        cleaned = clean_text(raw, preserve_case=True)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertIn("India defeated South Africa", cleaned)

    def test_cricket_entities_preserved(self):
        text = "India scored 176/7 and Bumrah took 2/18 in 4 overs."
        cleaned = clean_text(text, preserve_case=True)
        self.assertIn("176/7", cleaned)
        self.assertIn("2/18", cleaned)
        self.assertIn("4 overs", cleaned)

    def test_tokenize_sentences(self):
        text = (
            "Virat Kohli scored 76 runs. Jasprit Bumrah bowled four magnificent overs. "
            "India lifted the trophy in Barbados!"
        )
        sentences = tokenize_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertIn("Virat Kohli scored 76 runs.", sentences[0])

    def test_tokenize_words_removes_stopwords(self):
        text = "The captain and the team were very happy with the victory."
        words = tokenize_words(text, remove_stops=True, lowercase=True)
        self.assertNotIn("the", words)
        self.assertNotIn("and", words)
        self.assertIn("captain", words)
        self.assertIn("victory", words)

    def test_preprocess_for_vectorizer(self):
        text = "<b>India's</b> bowling attack was extraordinary in the 2024 final!"
        processed = preprocess_for_vectorizer(text)
        self.assertIsInstance(processed, str)
        self.assertIn("india", processed)
        self.assertIn("bowling", processed)
        self.assertIn("final", processed)

    def test_empty_and_short_inputs(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(tokenize_sentences(""), [])
        self.assertEqual(tokenize_words(""), [])
        self.assertFalse(is_valid_article("Too short"))
        self.assertTrue(is_valid_article("This is a sufficiently long cricket article with many words to test validity properly."))


if __name__ == "__main__":
    unittest.main()
