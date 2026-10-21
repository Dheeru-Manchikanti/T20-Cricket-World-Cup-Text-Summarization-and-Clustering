"""Unit tests for extractive text summarization."""

import unittest
from src.summarizer import summarize_text


class TestSummarizer(unittest.TestCase):
    def setUp(self):
        self.sample_text = (
            "India won the ICC Men's T20 World Cup 2024 by defeating South Africa by seven runs in Barbados. "
            "Virat Kohli anchored the innings with a brilliant 76 off 59 balls after India were 34 for 3. "
            "Axar Patel provided valuable acceleration with 47 runs off 31 deliveries. "
            "Heinrich Klaasen blasted 52 off 27 deliveries to keep South Africa firmly in the run chase. "
            "Jasprit Bumrah bowled two game-changing overs at the death, finishing with 2 for 18. "
            "Hardik Pandya defended 16 runs in the final over, aided by Suryakumar Yadav's miraculous boundary catch."
        )

    def test_summarize_normal_article(self):
        res = summarize_text(self.sample_text, max_sentences=2)
        self.assertIn("summary", res)
        self.assertEqual(res["summary_sentences"], 2)
        self.assertGreater(res["original_word_count"], res["summary_word_count"])
        self.assertGreater(res["reduction_percentage"], 30.0)

    def test_summarize_sentence_count_adjustments(self):
        res2 = summarize_text(self.sample_text, max_sentences=2)
        res3 = summarize_text(self.sample_text, max_sentences=3)
        self.assertEqual(res2["summary_sentences"], 2)
        self.assertEqual(res3["summary_sentences"], 3)
        self.assertLessEqual(res2["summary_word_count"], res3["summary_word_count"])

    def test_short_article_handling(self):
        short_text = "India defeated Pakistan by six runs in New York."
        res = summarize_text(short_text, max_sentences=2)
        self.assertEqual(res["summary"], short_text)
        self.assertEqual(res["summary_sentences"], 1)

    def test_empty_article_handling(self):
        res = summarize_text("", max_sentences=2)
        self.assertEqual(res["summary_word_count"], 0)
        self.assertEqual(res["reduction_percentage"], 0.0)

    def test_avoid_repeating_sentences(self):
        repeated_text = (
            "Jasprit Bumrah took two wickets in the final over. "
            "Jasprit Bumrah took two wickets in the final over. "
            "Virat Kohli scored 76 runs to guide India. "
            "Hardik Pandya defended the total."
        )
        res = summarize_text(repeated_text, max_sentences=2)
        # Should not include identical sentences twice
        summary = res["summary"]
        count = summary.count("Jasprit Bumrah took two wickets in the final over")
        self.assertLessEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
