# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

from typing import List, Dict, Tuple, Optional
import string
import emoji
import re

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS, NEGATIONS, AMPLIFIERS, EMOJI_SCORES, SIGNAL_WEIGHTS, LABEL_THRESHOLDS, WordSignal


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as dicts mapping word -> weight for fast lookup.
        self.positive_words = {w.lower(): weight for w, weight in positive_words.items()}
        self.negative_words = {w.lower(): weight for w, weight in negative_words.items()}

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        """
        padded_text = ''.join(f' {ch} ' if emoji.is_emoji(ch) else f'{ch}' for ch in text)
        cleaned = ''.join(ch for ch in padded_text if ch not in string.punctuation).strip().lower()
        final = re.sub(r'(.)(\1{2,})', r'\1\1', cleaned)


        print(f"Cleaned: {final}")    
        
        tokens = final.split()
        print(f"tokens: {tokens}")

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.
        """
       
        tokens = self.preprocess(text)
        score = 50

        # Modifier state
        amplifier = 1.0
        negated = False

        for token in tokens:
            if token in NEGATIONS:
                negated = True

            elif token in AMPLIFIERS:
                amplifier = min(amplifier * AMPLIFIERS[token], 3.0)

            elif token in EMOJI_SCORES:
                score += EMOJI_SCORES[token]
                amplifier = 1.0
                negated = False

            elif token in self.positive_words:
                weight = SIGNAL_WEIGHTS[WordSignal(self.positive_words[token]).name]
                change = weight * min(amplifier, 3.0)
                score += -change if negated else change
                amplifier = 1.0
                negated = False

            elif token in self.negative_words:
                weight = SIGNAL_WEIGHTS[WordSignal(self.negative_words[token]).name]
                change = weight * min(amplifier, 3.0)
                score += change if negated else -change
                amplifier = 1.0
                negated = False

        return int(score)


    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"
        """
        score = self.score_text(text)

        if score >= LABEL_THRESHOLDS["positive_above"]:
            return "positive"
        elif score <= LABEL_THRESHOLDS["negative_below"]:
            return "negative"
        else:
            return "mixed"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def analyze(self, text: str) -> dict:
        score = self.score_text(text)
        label = self.predict_label(text)
        explanation = self.explain(text)
        return {"text": text, "label": label, "score": score, "explanation": explanation}

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
