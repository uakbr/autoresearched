"""
Emotional-Learning evaluation harness. DO NOT MODIFY.

This is the equivalent of autoresearch's prepare.py.
It contains the fixed test dataset, evaluation function, and data utilities.
The experiment loop agent must NEVER edit this file.
"""

import sys
import os
import hashlib
import json

# Ensure backend/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score

from dataset import SAMPLE_POSTS, TRUE_LABELS, UNLABELED_EXAMPLES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIME_BUDGET = 30          # seconds per experiment (scikit-learn is fast on M4 Max)
RANDOM_SEED = 42
CLASSES = ["positive", "negative", "neutral", "mixed"]

# ---------------------------------------------------------------------------
# Fixed Test Dataset — 60 hand-labeled examples, 15 per class
# ---------------------------------------------------------------------------
# These examples are the ground truth for all experiments.
# Changing them invalidates every result in results.tsv.
#
# Sources:
#   - 30 drawn from UNLABELED_EXAMPLES in dataset.py (hand-labeled here)
#   - 30 additional curated examples targeting edge cases
#
# Difficulty tiers: easy / medium / hard (including sarcasm, slang, subtle)

TEST_TEXTS = [
    # ── POSITIVE (15) ────────────────────────────────────────────────────────

    # Easy positive (5)
    "Just got promoted at work, honestly couldn't be happier right now",
    "We won the championship game last night 🎉",
    "My little sister graduated today, couldn't be more proud 😊",
    "This made my day, thank you so much",
    "Best meal I have had in a long time",

    # Medium positive (5)
    "Ran my first 5K without stopping today",
    "Finally understand recursion after three weeks of confusion",
    "Reached out to someone I fell off with and they responded warmly",
    "Nailed the presentation, boss was impressed",
    "Feeling grateful for the people in my life",

    # Hard positive (5)
    "My friend remembered something small I mentioned months ago, meant a lot",
    "Went to therapy today and left feeling lighter",
    "The weather is gorgeous and I feel alive",
    "Everything just clicked into place today",
    "I genuinely enjoy coming to work every morning",

    # ── NEGATIVE (15) ────────────────────────────────────────────────────────

    # Easy negative (5)
    "Failed my driving test for the third time in a row",
    "My landlord is raising my rent again, I can't afford this",
    "Got rejected from every internship I applied to",
    "I cannot believe how rude that cashier was",
    "Nothing ever works out the way I want it to",

    # Medium negative (5)
    "My team lead took credit for my project in front of everyone",
    "My anxiety has been really bad lately, not sure what triggered it",
    "Wasted my entire weekend on something pointless",
    "I feel completely invisible at school",
    "Nobody even noticed I was gone for a week",

    # Hard negative — sarcasm (5)
    "Oh great, another all-hands meeting at 8am on a Friday 🙃",
    "Sure, I love being the only one who does any work on the group project",
    "Love how every update somehow makes the app worse",
    "Wow so nice of them to cancel last minute again",
    "Thanks for letting me know after the deadline passed",

    # ── NEUTRAL (15) ─────────────────────────────────────────────────────────

    # Easy neutral (5)
    "Went to the grocery store and grabbed some stuff for the week",
    "Had leftovers from yesterday for dinner",
    "The library closes at nine on Sundays",
    "The train arrived at exactly 3:15",
    "I have a dentist appointment next Tuesday",

    # Medium neutral (5)
    "Took a different route home and it added five minutes",
    "The software update finished installing overnight",
    "Watched a documentary about penguins last night",
    "We switched to a new project management tool at work",
    "The store was closed so I went to the one on Fifth Street",

    # Hard neutral (5)
    "I usually get there around 8:30",
    "My phone battery lasts about a day and a half",
    "Class starts at ten on Mondays and Wednesdays",
    "I ordered the same thing I always get",
    "There are about thirty people in my section",

    # ── MIXED (15) ───────────────────────────────────────────────────────────

    # Easy mixed (5)
    "Glad the project is over but honestly I learned nothing from it",
    "I love my job but the commute is killing me slowly",
    "Got the apartment I wanted but I'm terrified of living alone",
    "Excited for the trip but dreading the flight",
    "The feedback was constructive but honestly it stung a little",

    # Medium mixed (5)
    "Got the job offer but the salary is lower than I was hoping",
    "It's my birthday and I'm weirdly sad about it",
    "Won the argument but now I feel guilty about it",
    "New job starts Monday and I am equal parts thrilled and terrified",
    "The party was incredible but now I have to clean everything up",

    # Hard mixed (5)
    "Had a good cry tonight, needed it I think",
    "Things are looking up but I don't want to jinx it",
    "I passed the test but just barely",
    "Getting better at guitar but my fingers are killing me",
    "Finally moved to a new city, exciting but so lonely",
]

TEST_LABELS = [
    # Positive (15)
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    # Negative (15)
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative", "negative",
    # Neutral (15)
    "neutral", "neutral", "neutral", "neutral", "neutral",
    "neutral", "neutral", "neutral", "neutral", "neutral",
    "neutral", "neutral", "neutral", "neutral", "neutral",
    # Mixed (15)
    "mixed", "mixed", "mixed", "mixed", "mixed",
    "mixed", "mixed", "mixed", "mixed", "mixed",
    "mixed", "mixed", "mixed", "mixed", "mixed",
]

assert len(TEST_TEXTS) == 60, f"Expected 60 test examples, got {len(TEST_TEXTS)}"
assert len(TEST_LABELS) == 60, f"Expected 60 test labels, got {len(TEST_LABELS)}"

# Fast lookup set for filtering the unlabeled pool
_TEST_TEXTS_SET = frozenset(TEST_TEXTS)

# Integrity checksum — computed once, verified every evaluation
_CHECKSUM_DATA = json.dumps(
    [{"text": t, "label": l} for t, l in zip(TEST_TEXTS, TEST_LABELS)],
    sort_keys=True,
)
_EXPECTED_CHECKSUM = hashlib.sha256(_CHECKSUM_DATA.encode()).hexdigest()


def _verify_integrity():
    """Verify the test dataset has not been tampered with."""
    data = json.dumps(
        [{"text": t, "label": l} for t, l in zip(TEST_TEXTS, TEST_LABELS)],
        sort_keys=True,
    )
    actual = hashlib.sha256(data.encode()).hexdigest()
    assert actual == _EXPECTED_CHECKSUM, (
        f"Test dataset integrity check failed!\n"
        f"Expected: {_EXPECTED_CHECKSUM}\n"
        f"Got:      {actual}\n"
        f"The test set must NEVER change between experiments."
    )


# ---------------------------------------------------------------------------
# Evaluation Function — DO NOT MODIFY
# ---------------------------------------------------------------------------

def evaluate(predictions: list, labels: list = None) -> dict:
    """
    Fixed evaluation function. DO NOT MODIFY.

    Computes metrics for a list of predictions against the fixed test labels.

    Args:
        predictions: list of predicted labels (must be from CLASSES)
        labels: override test labels (default: TEST_LABELS). Only for testing.

    Returns:
        dict with:
            macro_f1   - PRIMARY metric (higher is better)
            accuracy   - secondary metric
            per_class  - {class_name: {precision, recall, f1, support}}
    """
    _verify_integrity()

    if labels is None:
        labels = TEST_LABELS

    assert len(predictions) == len(labels), (
        f"predictions length ({len(predictions)}) != labels length ({len(labels)})"
    )

    # Sanitize predictions: invalid labels become "mixed" (penalized naturally)
    valid = set(CLASSES)
    sanitized = [p if p in valid else "mixed" for p in predictions]

    # Primary metric: macro-averaged F1
    macro_f1 = f1_score(labels, sanitized, labels=CLASSES, average="macro", zero_division=0)

    # Secondary metric: accuracy
    accuracy = accuracy_score(labels, sanitized)

    # Per-class breakdown
    prec, rec, f1, support = precision_recall_fscore_support(
        labels, sanitized, labels=CLASSES, zero_division=0,
    )
    per_class = {}
    for i, cls in enumerate(CLASSES):
        per_class[cls] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Results Printing — greppable output format
# ---------------------------------------------------------------------------

def print_results(ml_metrics: dict, rb_metrics: dict, elapsed: float) -> None:
    """
    Print results in a greppable format matching autoresearch conventions.

    Extraction: grep "^macro_f1:" run.log
    """
    print("---")
    print(f"macro_f1:         {ml_metrics['macro_f1']:.6f}")
    print(f"accuracy:         {ml_metrics['accuracy']:.6f}")
    print(f"rb_macro_f1:      {rb_metrics['macro_f1']:.6f}")
    print(f"rb_accuracy:      {rb_metrics['accuracy']:.6f}")
    for cls in CLASSES:
        ml_f1 = ml_metrics["per_class"][cls]["f1"]
        print(f"{cls}_f1:     {ml_f1:.6f}")
    print(f"elapsed_seconds:  {elapsed:.1f}")


# ---------------------------------------------------------------------------
# Data Utilities
# ---------------------------------------------------------------------------

def get_seed_training_data() -> tuple:
    """
    Return the seed labeled training data from dataset.py.
    Returns (texts: list[str], labels: list[str]) — 10 examples.
    """
    return list(SAMPLE_POSTS), list(TRUE_LABELS)


def get_unlabeled_pool() -> list:
    """
    Return unlabeled examples from dataset.py, EXCLUDING any that appear
    in the fixed test set. This prevents train/test leakage.
    """
    return [t for t in UNLABELED_EXAMPLES if t not in _TEST_TEXTS_SET]


def get_test_data() -> tuple:
    """
    Return the fixed held-out test set.
    Returns (texts: list[str], labels: list[str]) — 60 examples.
    """
    return list(TEST_TEXTS), list(TEST_LABELS)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _verify_integrity()
    texts, labels = get_test_data()
    pool = get_unlabeled_pool()
    seed_t, seed_l = get_seed_training_data()

    print(f"Test set:       {len(texts)} examples")
    print(f"Seed training:  {len(seed_t)} examples")
    print(f"Unlabeled pool: {len(pool)} examples (after excluding test set)")

    # Verify class balance
    from collections import Counter
    dist = Counter(labels)
    print(f"Class distribution: {dict(dist)}")

    # Verify no overlap between test and training
    overlap = set(texts) & set(seed_t)
    assert not overlap, f"LEAKAGE: test/train overlap: {overlap}"

    # Verify no overlap between test and unlabeled pool
    pool_overlap = set(texts) & set(pool)
    assert not pool_overlap, f"LEAKAGE: test/pool overlap: {pool_overlap}"

    print("All integrity checks passed.")
