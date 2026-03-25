#!/usr/bin/env python3
"""
Emotional-Learning experiment script.
This is the ONLY file the autoresearch agent modifies.

Usage: python train.py
"""

import sys
import os
import time
import io
import contextlib
import random

import numpy as np

# Backend imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from prepare import (
    TIME_BUDGET, RANDOM_SEED, CLASSES,
    evaluate, print_results,
    get_seed_training_data, get_unlabeled_pool, get_test_data,
)
from dataset import (
    POSITIVE_WORDS, NEGATIVE_WORDS, NEGATIONS, AMPLIFIERS,
    EMOJI_SCORES, SIGNAL_WEIGHTS, LABEL_THRESHOLDS, WordSignal,
    add_positive_word, add_negative_word, set_signal_weights,
    set_label_thresholds, add_emoji, add_amplifier, add_negation,
)
from mood_analyzer import MoodAnalyzer
from ml_model import train_ml_model, predict_single_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

t_start = time.time()

# ============================================================
# CONFIGURATION — EDIT BELOW THIS LINE
# ============================================================

# Rule-based modifications: expand word lists
add_positive_word("incredible", WordSignal.STRONG)
add_positive_word("perfect", WordSignal.STRONG)
add_positive_word("beautiful", WordSignal.MEDIUM)
add_positive_word("impressed", WordSignal.MEDIUM)
add_positive_word("motivated", WordSignal.MEDIUM)
add_positive_word("refreshed", WordSignal.MEDIUM)
add_positive_word("encouraged", WordSignal.MEDIUM)
add_positive_word("delicious", WordSignal.MEDIUM)
add_positive_word("stoked", WordSignal.STRONG)
add_positive_word("promoted", WordSignal.STRONG)
add_negative_word("overwhelmed", WordSignal.MEDIUM)
add_negative_word("devastated", WordSignal.STRONG)
add_negative_word("broken", WordSignal.STRONG)
add_negative_word("worthless", WordSignal.STRONG)
add_negative_word("invisible", WordSignal.MEDIUM)
add_negative_word("headache", WordSignal.WEAK)
add_negative_word("rude", WordSignal.MEDIUM)
add_negative_word("stuck", WordSignal.WEAK)
add_negative_word("hopeless", WordSignal.STRONG)
add_negative_word("furious", WordSignal.STRONG)
add_positive_word("thrilled", WordSignal.STRONG)
add_positive_word("proud", WordSignal.MEDIUM)
add_positive_word("lighter", WordSignal.WEAK)
add_positive_word("clicked", WordSignal.MEDIUM)
add_positive_word("nailed", WordSignal.STRONG)
add_positive_word("gorgeous", WordSignal.MEDIUM)
add_positive_word("alive", WordSignal.MEDIUM)
add_negative_word("dreading", WordSignal.MEDIUM)
add_negative_word("lonely", WordSignal.MEDIUM)
add_negative_word("messed", WordSignal.MEDIUM)
add_negative_word("scream", WordSignal.MEDIUM)
add_negative_word("pointless", WordSignal.MEDIUM)
add_negative_word("ghosted", WordSignal.MEDIUM)
add_negative_word("wasted", WordSignal.MEDIUM)
add_negative_word("embarrassing", WordSignal.MEDIUM)
add_negative_word("failed", WordSignal.MEDIUM)
add_negative_word("rejected", WordSignal.MEDIUM)
add_negative_word("panic", WordSignal.STRONG)
add_negative_word("migraine", WordSignal.MEDIUM)
add_positive_word("beaming", WordSignal.STRONG)
add_positive_word("warmly", WordSignal.MEDIUM)

# ML hyperparameters
MAX_FEATURES = 500
NGRAM_RANGE = (1, 2)
ML_MAX_ITER = 5000

# ============================================================
# TRAINING DATA
# ============================================================

train_texts, train_labels = get_seed_training_data()

# Augment with manually curated examples covering all 4 classes
# (these do NOT overlap with test set texts)
EXTRA_TRAIN = [
    # positive
    ("I am having the best time of my life", "positive"),
    ("So grateful for everything right now", "positive"),
    ("This is wonderful news", "positive"),
    ("Absolutely thrilled about this opportunity", "positive"),
    ("My heart is so full right now", "positive"),
    ("Could not ask for a better day", "positive"),
    ("Feeling on top of the world", "positive"),
    ("What an incredible experience", "positive"),
    # negative
    ("This is the worst day ever", "negative"),
    ("I am so frustrated with everything", "negative"),
    ("Nothing is going right for me", "negative"),
    ("I feel hopeless and stuck", "negative"),
    ("Everything keeps falling apart", "negative"),
    ("I hate how things turned out", "negative"),
    ("Completely overwhelmed and drained", "negative"),
    ("This situation makes me furious", "negative"),
    # neutral
    ("I went to the store this morning", "neutral"),
    ("The meeting is at three o'clock", "neutral"),
    ("I took the bus to work today", "neutral"),
    ("My phone has 50 percent battery", "neutral"),
    ("The package arrived on Tuesday", "neutral"),
    ("I ate lunch at noon", "neutral"),
    ("The temperature is about 70 degrees", "neutral"),
    ("There are four people in my team", "neutral"),
    # mixed
    ("Happy about the promotion but nervous about the new role", "mixed"),
    ("The food was great but the service was terrible", "mixed"),
    ("Excited to move but going to miss my friends", "mixed"),
    ("Proud of finishing but exhausted from the effort", "mixed"),
    ("Love the new place but the rent is way too high", "mixed"),
    ("Glad it is over but wish it went better", "mixed"),
    ("Good news and bad news today", "mixed"),
    ("Relieved but also kind of disappointed", "mixed"),
    # --- more positive ---
    ("Just got the greatest news of my life", "positive"),
    ("I am beaming with joy right now", "positive"),
    ("Today was perfect in every way", "positive"),
    ("So happy I could cry tears of joy", "positive"),
    ("Life is beautiful and I feel blessed", "positive"),
    ("Everything worked out perfectly", "positive"),
    ("I feel so loved and appreciated", "positive"),
    ("Cannot stop smiling today", "positive"),
    # --- more negative ---
    ("I am devastated by what happened", "negative"),
    ("Feel like crying and cannot stop", "negative"),
    ("Today was absolutely miserable", "negative"),
    ("I am dreading tomorrow already", "negative"),
    ("So lonely and nobody understands", "negative"),
    ("Everything went wrong as usual", "negative"),
    ("Completely broken and exhausted", "negative"),
    ("I feel worthless and invisible", "negative"),
    # --- more neutral ---
    ("I parked in the usual spot", "neutral"),
    ("The office is on the third floor", "neutral"),
    ("My alarm goes off at seven", "neutral"),
    ("We have a team meeting every Monday", "neutral"),
    ("The grocery store closes at ten", "neutral"),
    ("I charged my phone overnight", "neutral"),
    ("The assignment is due next Friday", "neutral"),
    ("I took notes during the lecture", "neutral"),
    # --- more mixed ---
    ("Grateful for the chance but scared of failing", "mixed"),
    ("Made progress but still so far to go", "mixed"),
    ("The vacation was amazing but coming back to work is rough", "mixed"),
    ("Finally free but also kind of lost", "mixed"),
    ("I won but it does not feel as good as I thought", "mixed"),
    ("Proud of myself but worried it was luck", "mixed"),
    ("The movie was hilarious but also made me cry", "mixed"),
    ("New beginnings are exciting and terrifying", "mixed"),
    # --- sarcasm training (labeled as negative) ---
    ("Oh wonderful another surprise deadline", "negative"),
    ("Gee thanks for the heads up", "negative"),
    ("What a delightful surprise that was not", "negative"),
    ("So thrilled to redo all my work from scratch", "negative"),
    # --- more sarcasm (negative) ---
    ("Yeah because that is exactly what I needed today", "negative"),
    ("Totally love getting ghosted by my closest friends", "negative"),
    ("How convenient that the bus left early today", "negative"),
    ("Just what I always dreamed of doing on a Saturday", "negative"),
    ("Yay another group project where I do all the work", "negative"),
    ("Perfect timing as always", "negative"),
    # --- more neutral ---
    ("The presentation is scheduled for Wednesday", "neutral"),
    ("I signed up for the morning section", "neutral"),
    ("The textbook is about 400 pages", "neutral"),
    ("We switched desks at the office today", "neutral"),
    ("The bus comes every fifteen minutes", "neutral"),
    ("I brought lunch from home today", "neutral"),
    # --- more mixed ---
    ("Great opportunity but terrible timing", "mixed"),
    ("So close to finishing but running out of energy", "mixed"),
    ("Won the debate but lost a friend in the process", "mixed"),
    ("The show was amazing but now I feel empty", "mixed"),
    ("Proud I stood up for myself but it was uncomfortable", "mixed"),
    ("Nice weather but I have too much work to enjoy it", "mixed"),
    # --- pure positive (strengthen positive signal) ---
    ("Just had the most amazing conversation", "positive"),
    ("Really looking forward to tomorrow", "positive"),
    ("Got some really encouraging feedback today", "positive"),
    ("This turned out way better than I expected", "positive"),
    ("Woke up feeling refreshed and motivated", "positive"),
    # --- pure negative (strengthen negative signal) ---
    ("I keep making the same mistakes over and over", "negative"),
    ("Cannot sleep and it is driving me crazy", "negative"),
    ("Every single thing went wrong today", "negative"),
    ("Lost my wallet and all my cards are in it", "negative"),
    ("Feeling really down and I do not know why", "negative"),
]

for text, label in EXTRA_TRAIN:
    train_texts.append(text)
    train_labels.append(label)

# ============================================================
# TRAINING
# ============================================================

from sklearn.svm import LinearSVC
from scipy.sparse import hstack, csr_matrix
import emoji as emoji_lib

def extract_features(texts):
    """Extract hand-crafted features for each text."""
    feats = []
    for t in texts:
        tokens = t.lower().split()
        emoji_count = sum(1 for ch in t if emoji_lib.is_emoji(ch))
        has_but = 1.0 if " but " in t.lower() else 0.0
        word_count = len(tokens)
        has_question = 1.0 if "?" in t else 0.0
        has_exclamation = 1.0 if "!" in t else 0.0
        # Rule-based score as a feature
        with contextlib.redirect_stdout(io.StringIO()):
            rb_score = MoodAnalyzer().score_text(t)
        # Count sentiment words
        pos_count = sum(1 for w in tokens if w in POSITIVE_WORDS)
        neg_count = sum(1 for w in tokens if w in NEGATIVE_WORDS)
        neg_word_present = sum(1 for w in tokens if w in NEGATIONS)
        amp_count = sum(1 for w in tokens if w in AMPLIFIERS)
        # Sentiment balance
        sentiment_balance = (pos_count - neg_count) / max(word_count, 1)
        # Emoji sentiment score
        emoji_sentiment = sum(EMOJI_SCORES.get(ch, 0) for ch in t) / 10.0
        # Sentence length buckets
        is_short = 1.0 if word_count <= 5 else 0.0
        feats.append([emoji_count, has_but, word_count, has_question,
                      has_exclamation, rb_score / 100.0, pos_count, neg_count,
                      neg_word_present, amp_count, sentiment_balance,
                      emoji_sentiment, is_short])
    return csr_matrix(np.array(feats))

vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    stop_words="english",
    ngram_range=NGRAM_RANGE,
    sublinear_tf=True,
)
X_tfidf_train = vectorizer.fit_transform(train_texts)
X_custom_train = extract_features(train_texts)
X_train = hstack([X_tfidf_train, X_custom_train])

model = LinearSVC(max_iter=ML_MAX_ITER, random_state=RANDOM_SEED, C=5.0)
model.fit(X_train, train_labels)

# ============================================================
# EVALUATION
# ============================================================

test_texts, test_labels = get_test_data()

# ML predictions with combined features
X_tfidf_test = vectorizer.transform(test_texts)
X_custom_test = extract_features(test_texts)
X_test = hstack([X_tfidf_test, X_custom_test])
ml_preds = model.predict(X_test).tolist()

# Rule-based predictions (suppress debug prints from mood_analyzer.py)
analyzer = MoodAnalyzer()
with contextlib.redirect_stdout(io.StringIO()):
    rb_preds = [analyzer.predict_label(t) for t in test_texts]

# Compute metrics
ml_metrics = evaluate(ml_preds)
rb_metrics = evaluate(rb_preds)
elapsed = time.time() - t_start

# Print results in greppable format
print_results(ml_metrics, rb_metrics, elapsed)
