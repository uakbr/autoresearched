#!/usr/bin/env python3
"""
Emotional-Learning experiment script (v2 — rigorous three-way split).
This is the ONLY file the autoresearch agent modifies.

Uses prepare_v2.py evaluation harness with DEV / VAL / TEST splits,
cross-validation, and overfitting detection.

Usage: python train_v2.py
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

from prepare_v2 import (
    TIME_BUDGET, RANDOM_SEED, CLASSES,
    evaluate, print_results,
    get_seed_training_data, get_unlabeled_pool,
    get_dev_data, get_val_data, get_test_data,
    cross_validate,
)
from dataset import (
    POSITIVE_WORDS, NEGATIVE_WORDS, NEGATIONS, AMPLIFIERS,
    EMOJI_SCORES, SIGNAL_WEIGHTS, LABEL_THRESHOLDS, WordSignal,
    add_positive_word, add_negative_word, set_signal_weights,
    set_label_thresholds, add_emoji, add_amplifier, add_negation,
)
from mood_analyzer import MoodAnalyzer
from ml_model import train_ml_model, predict_single_text

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from scipy.sparse import hstack, csr_matrix
import emoji as emoji_lib

# Reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

t_start = time.time()

# ============================================================
# CONFIGURATION — EDIT BELOW THIS LINE
# ============================================================

# Rule-based modifications: adjust thresholds and expand word lists
set_label_thresholds(55, 45)  # Narrower mixed zone for more decisive predictions
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
# Achievement words
add_positive_word("won", WordSignal.STRONG)
add_positive_word("championship", WordSignal.STRONG)
add_positive_word("meant", WordSignal.WEAK)
add_positive_word("remembered", WordSignal.WEAK)
# Complaint/frustration words
add_negative_word("afford", WordSignal.MEDIUM)
add_negative_word("raising", WordSignal.WEAK)
# Amplifiers for emphasis
add_amplifier("genuinely", 1.5)
add_amplifier("truly", 1.5)
# Hedging words (suggest ambivalence/mixed)
add_positive_word("weirdly", WordSignal.WEAK)
# Narrative-positive words (quiet joy, achievement)
add_positive_word("surreal", WordSignal.MEDIUM)
add_positive_word("surprise", WordSignal.MEDIUM)
add_positive_word("complimented", WordSignal.MEDIUM)
add_positive_word("compliment", WordSignal.MEDIUM)
add_positive_word("peace", WordSignal.MEDIUM)
add_positive_word("sunrise", WordSignal.WEAK)
add_positive_word("invincible", WordSignal.STRONG)
add_positive_word("cried", WordSignal.WEAK)  # tears of joy context
add_positive_word("learned", WordSignal.WEAK)
add_positive_word("finished", WordSignal.WEAK)
add_positive_word("turned", WordSignal.WEAK)
add_positive_word("scratch", WordSignal.WEAK)  # "from scratch" = accomplishment
# Achievement/career positive words
add_positive_word("helpful", WordSignal.WEAK)
add_positive_word("loved", WordSignal.MEDIUM)
add_positive_word("better", WordSignal.WEAK)
add_positive_word("paid", WordSignal.WEAK)  # "paid off" = reward
add_positive_word("delicious", WordSignal.MEDIUM)
# Subtle negative words for complaints
add_negative_word("cracked", WordSignal.MEDIUM)
add_negative_word("wrong", WordSignal.WEAK)
add_negative_word("hungry", WordSignal.WEAK)
add_negative_word("skipped", WordSignal.WEAK)
add_negative_word("guilty", WordSignal.MEDIUM)
add_negative_word("toxic", WordSignal.STRONG)
add_negative_word("excuse", WordSignal.WEAK)
# Weaken ambiguous negative words (appear in positive outcomes too)
from dataset import change_word_weight
change_word_weight("worried", WordSignal.WEAK)

# ML hyperparameters
MAX_FEATURES = 400
NGRAM_RANGE = (1, 2)
ML_MAX_ITER = 5000

# ============================================================
# TRAINING DATA
# ============================================================

train_texts, train_labels = get_seed_training_data()

# Augment with manually curated examples covering all 4 classes
# (these do NOT overlap with dev, val, or test set texts)
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
    # exp4: neutral - change/alternative/routine patterns
    ("The coffee shop down the street just opened last week", "neutral"),
    ("My roommate rearranged the living room over the weekend", "neutral"),
    ("I set a reminder to pick up my prescription tomorrow", "neutral"),
    ("The elevator was broken so everyone used the stairs", "neutral"),
    ("The vending machine was empty so I went to the cafeteria", "neutral"),
    ("I normally drive but today I took the bus instead", "neutral"),
    ("The new restaurant opened on the corner of Main Street", "neutral"),
    ("My neighbor moved some boxes into the garage yesterday", "neutral"),
    ("I scheduled a dentist appointment for next month", "neutral"),
    ("The old machine broke down and they replaced it with a new one", "neutral"),
    ("They changed the menu at the cafeteria this week", "neutral"),
    ("I took a different elevator because the first one was full", "neutral"),
    ("The snack machine was out of chips so I got pretzels", "neutral"),
    ("My coworker switched desks with someone on the other side", "neutral"),
    ("The construction crew started working on the building next door", "neutral"),
    # exp6: positive-despite-worry (exam anxiety patterns)
    ("Got the highest score on the test I was so nervous about", "positive"),
    ("Passed the exam I was dreading and it feels so good", "positive"),
    ("Was worried about the results but I actually aced it", "positive"),
    # exp6: negative-despite-positive-words (deception/ghosting/penalty)
    ("Got stood up after what seemed like a great conversation", "negative"),
    ("Thought we had a great connection but they ghosted me", "negative"),
    ("Got a ticket right outside my building what a joke", "negative"),
    # exp3: narrative-positive (achievement/surprise/quiet joy patterns)
    ("Finally finished the project and it feels surreal", "positive"),
    ("Got a surprise gift and it made my whole day", "positive"),
    ("Someone complimented my work and I cannot stop smiling", "positive"),
    ("Woke up early and the sunrise was absolutely stunning", "positive"),
    ("Made it from scratch and it turned out perfectly", "positive"),
    ("Got the results back and I did way better than expected", "positive"),
    ("My friend showed up unannounced and it was the best surprise", "positive"),
    ("Watched the sunset from the rooftop in total peace", "positive"),
    ("Finally passed the test after studying for weeks", "positive"),
    ("The care package from my family brightened my entire week", "positive"),
    ("A stranger held the door and said something really kind", "positive"),
    ("Finished the book I started months ago and loved the ending", "positive"),
    ("My hard work finally paid off and I feel on cloud nine", "positive"),
    ("Cooked dinner for my friends and everyone asked for the recipe", "positive"),
    ("Got accepted into the program I have been dreaming about", "positive"),
    # exp19: targeted at specific dev errors
    ("Got the best score on the test I was worried about", "positive"),  # positive despite "worried"
    ("Flight got delayed and I missed everything I planned", "negative"),  # cancelled/missed = negative
    ("I usually walk but took the subway this time", "neutral"),  # neutral alternative with "but"
    ("The talk went okay I guess hard to say really", "mixed"),  # ambivalent/uncertain = mixed
]

for text, label in EXTRA_TRAIN:
    train_texts.append(text)
    train_labels.append(label)

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(texts):
    """Extract 15 hand-crafted features for each text."""
    feats = []
    for t in texts:
        tokens = t.lower().split()
        emoji_count = sum(1 for ch in t if emoji_lib.is_emoji(ch))
        has_but = 1.0 if " but " in t.lower() else 0.0
        word_count = len(tokens)
        has_question = 1.0 if "?" in t else 0.0
        has_exclamation = 1.0 if "!" in t else 0.0
        # Rule-based score as a feature (suppress debug prints)
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
        # avg_word_length as formality proxy
        avg_word_len = sum(len(w) for w in tokens) / max(len(tokens), 1) / 10.0
        # Sarcasm starter detection
        sarcasm_starters = {'oh', 'sure', 'wow', 'gee', 'yay', 'great', 'fantastic', 'thanks', 'love'}
        has_sarcasm_start = 1.0 if tokens and tokens[0] in sarcasm_starters else 0.0
        feats.append([emoji_count, has_but, word_count, has_question,
                      has_exclamation, rb_score / 100.0, pos_count, neg_count,
                      neg_word_present, amp_count, sentiment_balance,
                      emoji_sentiment, is_short, avg_word_len, has_sarcasm_start])
    return csr_matrix(np.array(feats))

# ============================================================
# TRAINING
# ============================================================

vectorizer = CountVectorizer(
    max_features=MAX_FEATURES,
    stop_words="english",
    ngram_range=NGRAM_RANGE,
    binary=True,  # presence/absence instead of counts
)
X_tfidf_train = vectorizer.fit_transform(train_texts)
X_custom_train = extract_features(train_texts)
X_train = hstack([X_tfidf_train, X_custom_train])

# Ensemble of 3 models with different C values
from collections import Counter as _Counter
model_c1 = LinearSVC(max_iter=ML_MAX_ITER, random_state=RANDOM_SEED, C=1.0)
model_c5 = LinearSVC(max_iter=ML_MAX_ITER, random_state=RANDOM_SEED, C=5.0)
model_c20 = LinearSVC(max_iter=ML_MAX_ITER, random_state=RANDOM_SEED, C=20.0)
model_c1.fit(X_train, train_labels)
model_c5.fit(X_train, train_labels)
model_c20.fit(X_train, train_labels)


# ============================================================
# PREDICTION HELPER
# ============================================================

def predict_cascade(texts):
    """Ensemble 3-model majority vote with neutral-rescue heuristic."""
    X_tfidf = vectorizer.transform(texts)
    X_custom = extract_features(texts)
    X = hstack([X_tfidf, X_custom])
    p1 = model_c1.predict(X).tolist()
    p5 = model_c5.predict(X).tolist()
    p20 = model_c20.predict(X).tolist()
    # Get decision function margins for confidence
    df1 = model_c1.decision_function(X)
    results = []
    classes_list = list(model_c1.classes_)
    for i in range(len(texts)):
        votes = [p1[i], p5[i], p20[i]]
        winner = _Counter(votes).most_common(1)[0][0]
        # Neutral rescue: override to neutral when language signals acceptance/hedging
        tokens = texts[i].lower().split()
        lower_text = texts[i].lower()
        acceptance_words = {'fine', 'needed', 'okay', 'exactly'}
        has_acceptance = any(w in tokens for w in acceptance_words)
        negated_neg = "didn't feel" in lower_text or "don't feel" in lower_text
        # Hedging phrase: "honestly it was fine" / "honestly + fine/okay"
        hedging = ("honestly" in tokens and ("fine" in tokens or "okay" in tokens))
        # Positive rescue: "which never happens" = pleasant surprise idiom
        pleasant_surprise = "never happens" in lower_text or "which never" in lower_text
        if winner == "negative":
            neg_idx = classes_list.index("negative") if "negative" in classes_list else -1
            if neg_idx >= 0:
                margin = df1[i][neg_idx]
                if has_acceptance and margin < 0.5:
                    winner = "neutral"
                elif negated_neg and margin < 1.5:
                    winner = "neutral"
                elif pleasant_surprise and margin < 0.3:
                    winner = "positive"
        elif winner == "positive" and hedging:
            pos_idx = classes_list.index("positive") if "positive" in classes_list else -1
            if pos_idx >= 0:
                margin = df1[i][pos_idx]
                if margin < 0.5:
                    winner = "neutral"
        results.append(winner)
    return results


# ============================================================
# CROSS-VALIDATION (on training data)
# ============================================================

class CascadeModel:
    """Wrapper for cross_validate compatibility: .fit(texts, labels) and .predict(texts)."""
    def __init__(self):
        self.vectorizer = None
        self.models = None

    def fit(self, texts, labels):
        self.vectorizer = CountVectorizer(
            max_features=MAX_FEATURES,
            stop_words="english",
            ngram_range=NGRAM_RANGE,
            binary=True,
        )
        X_tfidf = self.vectorizer.fit_transform(texts)
        X_custom = extract_features(texts)
        X = hstack([X_tfidf, X_custom])
        self.models = []
        for c_val in [1.0, 5.0, 20.0]:
            m = LinearSVC(max_iter=ML_MAX_ITER, random_state=RANDOM_SEED, C=c_val)
            m.fit(X, labels)
            self.models.append(m)

    def predict(self, texts):
        X_tfidf = self.vectorizer.transform(texts)
        X_custom = extract_features(texts)
        X = hstack([X_tfidf, X_custom])
        all_preds = [m.predict(X).tolist() for m in self.models]
        results = []
        for i in range(len(texts)):
            votes = [p[i] for p in all_preds]
            results.append(_Counter(votes).most_common(1)[0][0])
        return results


cv_mean, cv_std = cross_validate(
    build_model_fn=CascadeModel,
    texts=train_texts,
    labels=train_labels,
    k=5,
)

# ============================================================
# EVALUATION
# ============================================================

# DEV set — used for keep/discard decisions
dev_texts, dev_labels = get_dev_data()
dev_preds = predict_cascade(dev_texts)
dev_metrics = evaluate(dev_preds, dev_labels)

# VAL set — used for overfitting detection (reported but not used for keep/discard)
val_texts, val_labels = get_val_data()
val_preds = predict_cascade(val_texts)
val_metrics = evaluate(val_preds, val_labels)

elapsed = time.time() - t_start

# Print results in greppable format
print_results(dev_metrics, val_metrics, cv_mean, cv_std, elapsed)

