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

# Rule-based modifications (baseline: no changes)
# e.g.: add_positive_word("stoked", WordSignal.STRONG)
# e.g.: set_label_thresholds(55, 45)

# ML hyperparameters
MAX_FEATURES = 500
NGRAM_RANGE = (1, 2)
ML_MAX_ITER = 1000

# ============================================================
# TRAINING DATA
# ============================================================

train_texts, train_labels = get_seed_training_data()

# ============================================================
# TRAINING
# ============================================================

vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    stop_words="english",
    ngram_range=NGRAM_RANGE,
    sublinear_tf=True,
)
X_train = vectorizer.fit_transform(train_texts)

model = LogisticRegression(max_iter=ML_MAX_ITER, random_state=RANDOM_SEED)
model.fit(X_train, train_labels)

# ============================================================
# EVALUATION
# ============================================================

test_texts, test_labels = get_test_data()

# ML predictions
X_test = vectorizer.transform(test_texts)
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
