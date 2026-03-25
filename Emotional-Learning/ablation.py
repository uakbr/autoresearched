#!/usr/bin/env python3
"""
Ablation study for Emotional-Learning autoresearch v2.

Removes one component at a time from the full model and measures impact.

Usage: python ablation.py
"""

import sys
import os
import io
import contextlib
import random
import copy

import numpy as np

# Backend imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from prepare import (
    RANDOM_SEED, CLASSES,
    evaluate,
    get_seed_training_data, get_test_data,
)
from dataset import (
    POSITIVE_WORDS, NEGATIVE_WORDS, NEGATIONS, AMPLIFIERS,
    EMOJI_SCORES, SIGNAL_WEIGHTS, LABEL_THRESHOLDS, WordSignal,
    add_positive_word, add_negative_word, set_signal_weights,
    set_label_thresholds, add_amplifier, add_negation,
)
from mood_analyzer import MoodAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from scipy.sparse import hstack, csr_matrix
import emoji as emoji_lib

# Reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# EXTRA_TRAIN — duplicated from train.py
# ============================================================

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
    # Help stage 1 neutral detector
    ("It takes about twenty minutes to get there", "neutral"),
    ("The test is next Thursday", "neutral"),
    ("I grabbed a coffee on the way in", "neutral"),
    ("There are two sections this semester", "neutral"),
    # Neutral: factual changes (prevent stage 2 from mislabeling as negative)
    ("We moved to a different tool at the office", "neutral"),
    ("The store was closed so I tried another one", "neutral"),
    ("They switched the schedule around this week", "neutral"),
    # exp121: achievement/event narratives as positive
    ("We won the match and everyone celebrated", "positive"),
    ("Completed my first marathon without walking", "positive"),
    # exp121: complaint narratives as negative
    ("They took credit for everything I did", "negative"),
    ("Prices keep going up and my pay stays the same", "negative"),
    # exp121: bittersweet birthday/cry patterns as mixed
    ("My birthday made me happy and sad at the same time", "mixed"),
    ("Had a good long cry and honestly it helped", "mixed"),
    # exp127: physical achievement (positive) + pessimism (negative)
    ("Ran ten miles without stopping once", "positive"),
    ("Nothing ever goes my way no matter what I do", "negative"),
    # exp128: aha-moment positive, alternative-neutral, birthday-mixed
    ("After weeks of confusion it finally all makes sense", "positive"),
    ("The place was closed so I went somewhere else", "neutral"),
    ("My birthday was fun but also kind of sad", "mixed"),
    # exp129: reconnection as positive
    ("Reached out to an old friend and they were so happy to hear from me", "positive"),
    ("Texted someone I lost touch with and they responded right away", "positive"),
    # exp130: mundane routine neutral (prevent stage 1 from calling emotional)
    ("My phone battery usually lasts all day", "neutral"),
    ("I got the same order I always get", "neutral"),
    ("The usual spot was taken so I sat somewhere else", "neutral"),
]


def _get_full_training_data():
    """Return seed + EXTRA_TRAIN combined."""
    train_texts, train_labels = get_seed_training_data()
    for text, label in EXTRA_TRAIN:
        train_texts.append(text)
        train_labels.append(label)
    return train_texts, train_labels


# ============================================================
# Apply word-list expansions (same as train.py)
# ============================================================

def _apply_word_list_expansions():
    """Apply all add_positive_word / add_negative_word calls from train.py."""
    set_label_thresholds(55, 45)
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
    add_negative_word("rude", WordSignal.MEDIUM)
    add_negative_word("cancel", WordSignal.WEAK)
    add_negative_word("deadline", WordSignal.WEAK)
    add_negative_word("worse", WordSignal.MEDIUM)
    add_positive_word("won", WordSignal.STRONG)
    add_positive_word("championship", WordSignal.STRONG)
    add_positive_word("meant", WordSignal.WEAK)
    add_positive_word("remembered", WordSignal.WEAK)
    add_negative_word("afford", WordSignal.MEDIUM)
    add_negative_word("raising", WordSignal.WEAK)
    add_amplifier("genuinely", 1.5)
    add_amplifier("truly", 1.5)
    add_positive_word("weirdly", WordSignal.WEAK)


# ============================================================
# Feature extraction helpers
# ============================================================

def extract_features_full(texts):
    """Full 15-feature extraction (same as train.py)."""
    feats = []
    for t in texts:
        tokens = t.lower().split()
        emoji_count = sum(1 for ch in t if emoji_lib.is_emoji(ch))
        has_but = 1.0 if " but " in t.lower() else 0.0
        word_count = len(tokens)
        has_question = 1.0 if "?" in t else 0.0
        has_exclamation = 1.0 if "!" in t else 0.0
        with contextlib.redirect_stdout(io.StringIO()):
            rb_score = MoodAnalyzer().score_text(t)
        pos_count = sum(1 for w in tokens if w in POSITIVE_WORDS)
        neg_count = sum(1 for w in tokens if w in NEGATIVE_WORDS)
        neg_word_present = sum(1 for w in tokens if w in NEGATIONS)
        amp_count = sum(1 for w in tokens if w in AMPLIFIERS)
        sentiment_balance = (pos_count - neg_count) / max(word_count, 1)
        emoji_sentiment = sum(EMOJI_SCORES.get(ch, 0) for ch in t) / 10.0
        is_short = 1.0 if word_count <= 5 else 0.0
        avg_word_len = sum(len(w) for w in tokens) / max(len(tokens), 1) / 10.0
        sarcasm_starters = {'oh', 'sure', 'wow', 'gee', 'yay', 'great', 'fantastic', 'thanks', 'love'}
        has_sarcasm_start = 1.0 if tokens and tokens[0] in sarcasm_starters else 0.0
        feats.append([emoji_count, has_but, word_count, has_question,
                      has_exclamation, rb_score / 100.0, pos_count, neg_count,
                      neg_word_present, amp_count, sentiment_balance,
                      emoji_sentiment, is_short, avg_word_len, has_sarcasm_start])
    return csr_matrix(np.array(feats))


def extract_features_no_rb_score(texts):
    """All features EXCEPT rb_score (feature index 5)."""
    feats = []
    for t in texts:
        tokens = t.lower().split()
        emoji_count = sum(1 for ch in t if emoji_lib.is_emoji(ch))
        has_but = 1.0 if " but " in t.lower() else 0.0
        word_count = len(tokens)
        has_question = 1.0 if "?" in t else 0.0
        has_exclamation = 1.0 if "!" in t else 0.0
        pos_count = sum(1 for w in tokens if w in POSITIVE_WORDS)
        neg_count = sum(1 for w in tokens if w in NEGATIVE_WORDS)
        neg_word_present = sum(1 for w in tokens if w in NEGATIONS)
        amp_count = sum(1 for w in tokens if w in AMPLIFIERS)
        sentiment_balance = (pos_count - neg_count) / max(word_count, 1)
        emoji_sentiment = sum(EMOJI_SCORES.get(ch, 0) for ch in t) / 10.0
        is_short = 1.0 if word_count <= 5 else 0.0
        avg_word_len = sum(len(w) for w in tokens) / max(len(tokens), 1) / 10.0
        sarcasm_starters = {'oh', 'sure', 'wow', 'gee', 'yay', 'great', 'fantastic', 'thanks', 'love'}
        has_sarcasm_start = 1.0 if tokens and tokens[0] in sarcasm_starters else 0.0
        feats.append([emoji_count, has_but, word_count, has_question,
                      has_exclamation, pos_count, neg_count,
                      neg_word_present, amp_count, sentiment_balance,
                      emoji_sentiment, is_short, avg_word_len, has_sarcasm_start])
    return csr_matrix(np.array(feats))


def extract_features_no_sarcasm(texts):
    """All features EXCEPT has_sarcasm_start (feature index 14)."""
    feats = []
    for t in texts:
        tokens = t.lower().split()
        emoji_count = sum(1 for ch in t if emoji_lib.is_emoji(ch))
        has_but = 1.0 if " but " in t.lower() else 0.0
        word_count = len(tokens)
        has_question = 1.0 if "?" in t else 0.0
        has_exclamation = 1.0 if "!" in t else 0.0
        with contextlib.redirect_stdout(io.StringIO()):
            rb_score = MoodAnalyzer().score_text(t)
        pos_count = sum(1 for w in tokens if w in POSITIVE_WORDS)
        neg_count = sum(1 for w in tokens if w in NEGATIVE_WORDS)
        neg_word_present = sum(1 for w in tokens if w in NEGATIONS)
        amp_count = sum(1 for w in tokens if w in AMPLIFIERS)
        sentiment_balance = (pos_count - neg_count) / max(word_count, 1)
        emoji_sentiment = sum(EMOJI_SCORES.get(ch, 0) for ch in t) / 10.0
        is_short = 1.0 if word_count <= 5 else 0.0
        avg_word_len = sum(len(w) for w in tokens) / max(len(tokens), 1) / 10.0
        feats.append([emoji_count, has_but, word_count, has_question,
                      has_exclamation, rb_score / 100.0, pos_count, neg_count,
                      neg_word_present, amp_count, sentiment_balance,
                      emoji_sentiment, is_short, avg_word_len])
    return csr_matrix(np.array(feats))


# ============================================================
# ABLATION VARIANTS
# ============================================================

def _run_cascade(train_texts, train_labels, test_texts, vectorizer_cls,
                 vectorizer_kwargs, feature_fn, use_cascade=True):
    """
    Generic cascade runner parameterized by vectorizer and feature extractor.

    Args:
        vectorizer_cls: CountVectorizer or TfidfVectorizer
        vectorizer_kwargs: dict of kwargs for the vectorizer
        feature_fn: callable(texts) -> sparse matrix, or None for no custom features
        use_cascade: if True, use two-stage; if False, flat 4-class classification
    """
    vec = vectorizer_cls(**vectorizer_kwargs)
    X_bow_train = vec.fit_transform(train_texts)

    if feature_fn is not None:
        X_custom_train = feature_fn(train_texts)
        X_train = hstack([X_bow_train, X_custom_train])
    else:
        X_train = X_bow_train

    if use_cascade:
        # Stage 1: emotional vs neutral
        stage1_labels = ["neutral" if l == "neutral" else "emotional"
                         for l in train_labels]
        model_s1 = LinearSVC(max_iter=5000, random_state=42, C=5.0)
        model_s1.fit(X_train, stage1_labels)

        # Stage 2: positive vs negative vs mixed
        emo_idx = [i for i, l in enumerate(train_labels) if l != "neutral"]
        X_train_emo = X_train[emo_idx]
        emo_labels = [train_labels[i] for i in emo_idx]
        model_s2 = LinearSVC(max_iter=5000, random_state=42, C=5.0)
        model_s2.fit(X_train_emo, emo_labels)

        # Predict
        X_bow_test = vec.transform(test_texts)
        if feature_fn is not None:
            X_custom_test = feature_fn(test_texts)
            X_test = hstack([X_bow_test, X_custom_test])
        else:
            X_test = X_bow_test

        s1_preds = model_s1.predict(X_test).tolist()
        s2_preds = model_s2.predict(X_test).tolist()
        preds = []
        for i in range(len(test_texts)):
            if s1_preds[i] == "neutral":
                preds.append("neutral")
            else:
                preds.append(s2_preds[i])
        return preds
    else:
        # Flat 4-class classification
        model = LinearSVC(max_iter=5000, random_state=42, C=5.0)
        model.fit(X_train, train_labels)

        X_bow_test = vec.transform(test_texts)
        if feature_fn is not None:
            X_custom_test = feature_fn(test_texts)
            X_test = hstack([X_bow_test, X_custom_test])
        else:
            X_test = X_bow_test

        return model.predict(X_test).tolist()


# Default vectorizer kwargs
_DEFAULT_VEC_KWARGS = dict(
    max_features=300,
    stop_words="english",
    ngram_range=(1, 2),
    binary=True,
)


def ablation_full(train_texts, train_labels, test_texts):
    """Full model (baseline for comparison)."""
    return _run_cascade(train_texts, train_labels, test_texts,
                        CountVectorizer, _DEFAULT_VEC_KWARGS,
                        extract_features_full, use_cascade=True)


def ablation_no_rb_score(train_texts, train_labels, test_texts):
    """Remove rb_score feature."""
    return _run_cascade(train_texts, train_labels, test_texts,
                        CountVectorizer, _DEFAULT_VEC_KWARGS,
                        extract_features_no_rb_score, use_cascade=True)


def ablation_no_sarcasm(train_texts, train_labels, test_texts):
    """Remove has_sarcasm_start feature."""
    return _run_cascade(train_texts, train_labels, test_texts,
                        CountVectorizer, _DEFAULT_VEC_KWARGS,
                        extract_features_no_sarcasm, use_cascade=True)


def ablation_no_custom_features(train_texts, train_labels, test_texts):
    """Remove ALL custom features (CountVectorizer only)."""
    return _run_cascade(train_texts, train_labels, test_texts,
                        CountVectorizer, _DEFAULT_VEC_KWARGS,
                        None, use_cascade=True)


def ablation_no_cascade(train_texts, train_labels, test_texts):
    """Remove stage 1 (flat 4-class classification)."""
    return _run_cascade(train_texts, train_labels, test_texts,
                        CountVectorizer, _DEFAULT_VEC_KWARGS,
                        extract_features_full, use_cascade=False)


def ablation_no_word_expansion(train_texts, train_labels, test_texts):
    """
    Remove word list expansion: use original POSITIVE_WORDS/NEGATIVE_WORDS only.

    We save/restore the global dicts so the rest of the ablations are not affected.
    """
    # Save expanded state
    pos_backup = dict(POSITIVE_WORDS)
    neg_backup = dict(NEGATIVE_WORDS)
    amp_backup = dict(AMPLIFIERS)
    thresh_backup = dict(LABEL_THRESHOLDS)

    # Restore to original dataset.py defaults
    POSITIVE_WORDS.clear()
    POSITIVE_WORDS.update({
        "okay": WordSignal.WEAK, "fine": WordSignal.WEAK,
        "chill": WordSignal.WEAK, "hopeful": WordSignal.WEAK,
        "relaxed": WordSignal.WEAK, "happy": WordSignal.MEDIUM,
        "great": WordSignal.MEDIUM, "good": WordSignal.MEDIUM,
        "excited": WordSignal.MEDIUM, "fun": WordSignal.MEDIUM,
        "proud": WordSignal.MEDIUM, "enjoy": WordSignal.MEDIUM,
        "grateful": WordSignal.MEDIUM, "love": WordSignal.STRONG,
        "awesome": WordSignal.STRONG, "amazing": WordSignal.STRONG,
        "wonderful": WordSignal.STRONG, "fantastic": WordSignal.STRONG,
        "thriving": WordSignal.STRONG, "blessed": WordSignal.STRONG,
        "joy": WordSignal.STRONG, "hit": WordSignal.STRONG,
        "different": WordSignal.STRONG,
    })
    NEGATIVE_WORDS.clear()
    NEGATIVE_WORDS.update({
        "tired": WordSignal.WEAK, "boring": WordSignal.WEAK,
        "bad": WordSignal.WEAK, "sad": WordSignal.MEDIUM,
        "angry": WordSignal.MEDIUM, "upset": WordSignal.MEDIUM,
        "stressed": WordSignal.MEDIUM, "worried": WordSignal.MEDIUM,
        "anxious": WordSignal.MEDIUM, "lonely": WordSignal.MEDIUM,
        "frustrated": WordSignal.MEDIUM, "disappointed": WordSignal.MEDIUM,
        "terrible": WordSignal.STRONG, "awful": WordSignal.STRONG,
        "hate": WordSignal.STRONG, "crying": WordSignal.STRONG,
        "miserable": WordSignal.STRONG, "horrible": WordSignal.STRONG,
        "exhausted": WordSignal.STRONG, "dread": WordSignal.STRONG,
    })
    AMPLIFIERS.clear()
    AMPLIFIERS.update({
        "very": 1.5, "really": 1.5, "extremely": 2.0,
        "absolutely": 2.0, "so": 1.3, "quite": 1.2,
        "super": 1.5, "fr": 1.5,
    })
    LABEL_THRESHOLDS.clear()
    LABEL_THRESHOLDS.update({"positive_above": 60, "negative_below": 40})

    try:
        preds = _run_cascade(train_texts, train_labels, test_texts,
                             CountVectorizer, _DEFAULT_VEC_KWARGS,
                             extract_features_full, use_cascade=True)
    finally:
        # Restore expanded state
        POSITIVE_WORDS.clear()
        POSITIVE_WORDS.update(pos_backup)
        NEGATIVE_WORDS.clear()
        NEGATIVE_WORDS.update(neg_backup)
        AMPLIFIERS.clear()
        AMPLIFIERS.update(amp_backup)
        LABEL_THRESHOLDS.clear()
        LABEL_THRESHOLDS.update(thresh_backup)

    return preds


def ablation_tfidf_instead(train_texts, train_labels, test_texts):
    """Use TF-IDF instead of binary CountVectorizer."""
    tfidf_kwargs = dict(
        max_features=300,
        stop_words="english",
        ngram_range=(1, 2),
    )
    return _run_cascade(train_texts, train_labels, test_texts,
                        TfidfVectorizer, tfidf_kwargs,
                        extract_features_full, use_cascade=True)


def ablation_max_features_500(train_texts, train_labels, test_texts):
    """Use max_features=500 instead of 300."""
    kwargs_500 = dict(
        max_features=500,
        stop_words="english",
        ngram_range=(1, 2),
        binary=True,
    )
    return _run_cascade(train_texts, train_labels, test_texts,
                        CountVectorizer, kwargs_500,
                        extract_features_full, use_cascade=True)


# ============================================================
# MAIN — run ablation study and print table
# ============================================================

if __name__ == "__main__":
    # Apply word-list expansions to match train.py state
    _apply_word_list_expansions()

    # Load data
    train_texts, train_labels = _get_full_training_data()
    test_texts, test_labels = get_test_data()

    print(f"Training examples: {len(train_texts)}")
    print(f"Test examples:     {len(test_texts)}")
    print()

    # Define ablations: (name, description, function)
    ablations = [
        ("Full model",           "All components",                          ablation_full),
        ("- rb_score",           "Remove rule-based score feature",         ablation_no_rb_score),
        ("- has_sarcasm_start",  "Remove sarcasm starter feature",          ablation_no_sarcasm),
        ("- ALL custom feats",   "CountVectorizer only, no custom feats",   ablation_no_custom_features),
        ("- cascade (flat)",     "Flat 4-class instead of two-stage",       ablation_no_cascade),
        ("- word expansion",     "Original word lists only",                ablation_no_word_expansion),
        ("TF-IDF (not binary)",  "TF-IDF instead of binary CountVec",      ablation_tfidf_instead),
        ("max_features=500",     "500 instead of 300 max features",         ablation_max_features_500),
    ]

    # Run full model first to get baseline F1
    full_preds = ablation_full(train_texts, train_labels, test_texts)
    full_metrics = evaluate(full_preds)
    full_f1 = full_metrics["macro_f1"]

    # Run all ablations and collect results
    results = []
    for name, desc, fn in ablations:
        if name == "Full model":
            metrics = full_metrics
        else:
            preds = fn(train_texts, train_labels, test_texts)
            metrics = evaluate(preds)
        delta = metrics["macro_f1"] - full_f1
        results.append((name, desc, metrics, delta))

    # Print ablation table
    header = f"{'Ablation':<24} {'Description':<42} {'Macro-F1':>10} {'Delta':>8} {'Accuracy':>10}"
    sep = "-" * len(header)
    print("ABLATION STUDY")
    print(sep)
    print(header)
    print(sep)
    for name, desc, m, delta in results:
        delta_str = f"{delta:>+8.4f}" if name != "Full model" else f"{'---':>8}"
        print(f"{name:<24} {desc:<42} {m['macro_f1']:>10.4f} {delta_str} {m['accuracy']:>10.4f}")
    print(sep)

    # Per-class breakdown for full model
    print(f"\nFull model per-class breakdown:")
    pc = full_metrics["per_class"]
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for cls in CLASSES:
        c = pc[cls]
        print(f"  {cls:<12} {c['precision']:>10.4f} {c['recall']:>10.4f} {c['f1']:>10.4f} {c['support']:>10d}")

    # Summary
    print(f"\nFull model macro_f1: {full_f1:.4f}")
    worst_name, _, worst_m, worst_delta = min(results[1:], key=lambda r: r[3])
    print(f"Most impactful removal: {worst_name} (delta = {worst_delta:+.4f}, macro_f1 = {worst_m['macro_f1']:.4f})")
