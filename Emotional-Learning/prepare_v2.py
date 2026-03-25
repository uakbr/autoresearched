"""
Emotional-Learning evaluation harness v2 — rigorous three-way split.

This is the v2 evaluation harness for autoresearch.
It provides DEV / VALIDATION / TEST splits, cross-validation,
significance testing (McNemar), and strict integrity checks.

The experiment loop agent must NEVER edit this file.
"""

import sys
import os
import hashlib
import json
import warnings
from collections import Counter

# Ensure backend/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

from dataset import SAMPLE_POSTS, TRUE_LABELS, UNLABELED_EXAMPLES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIME_BUDGET = 30          # seconds per experiment
RANDOM_SEED = 42
CLASSES = ["positive", "negative", "neutral", "mixed"]

# ---------------------------------------------------------------------------
# TEST SET — 60 hand-labeled examples from prepare.py (verbatim, DO NOT EDIT)
# ---------------------------------------------------------------------------

TEST_TEXTS = [
    # -- POSITIVE (15) --
    "Just got promoted at work, honestly couldn't be happier right now",
    "We won the championship game last night \U0001f389",
    "My little sister graduated today, couldn't be more proud \U0001f60a",
    "This made my day, thank you so much",
    "Best meal I have had in a long time",
    "Ran my first 5K without stopping today",
    "Finally understand recursion after three weeks of confusion",
    "Reached out to someone I fell off with and they responded warmly",
    "Nailed the presentation, boss was impressed",
    "Feeling grateful for the people in my life",
    "My friend remembered something small I mentioned months ago, meant a lot",
    "Went to therapy today and left feeling lighter",
    "The weather is gorgeous and I feel alive",
    "Everything just clicked into place today",
    "I genuinely enjoy coming to work every morning",

    # -- NEGATIVE (15) --
    "Failed my driving test for the third time in a row",
    "My landlord is raising my rent again, I can't afford this",
    "Got rejected from every internship I applied to",
    "I cannot believe how rude that cashier was",
    "Nothing ever works out the way I want it to",
    "My team lead took credit for my project in front of everyone",
    "My anxiety has been really bad lately, not sure what triggered it",
    "Wasted my entire weekend on something pointless",
    "I feel completely invisible at school",
    "Nobody even noticed I was gone for a week",
    "Oh great, another all-hands meeting at 8am on a Friday \U0001f643",
    "Sure, I love being the only one who does any work on the group project",
    "Love how every update somehow makes the app worse",
    "Wow so nice of them to cancel last minute again",
    "Thanks for letting me know after the deadline passed",

    # -- NEUTRAL (15) --
    "Went to the grocery store and grabbed some stuff for the week",
    "Had leftovers from yesterday for dinner",
    "The library closes at nine on Sundays",
    "The train arrived at exactly 3:15",
    "I have a dentist appointment next Tuesday",
    "Took a different route home and it added five minutes",
    "The software update finished installing overnight",
    "Watched a documentary about penguins last night",
    "We switched to a new project management tool at work",
    "The store was closed so I went to the one on Fifth Street",
    "I usually get there around 8:30",
    "My phone battery lasts about a day and a half",
    "Class starts at ten on Mondays and Wednesdays",
    "I ordered the same thing I always get",
    "There are about thirty people in my section",

    # -- MIXED (15) --
    "Glad the project is over but honestly I learned nothing from it",
    "I love my job but the commute is killing me slowly",
    "Got the apartment I wanted but I'm terrified of living alone",
    "Excited for the trip but dreading the flight",
    "The feedback was constructive but honestly it stung a little",
    "Got the job offer but the salary is lower than I was hoping",
    "It's my birthday and I'm weirdly sad about it",
    "Won the argument but now I feel guilty about it",
    "New job starts Monday and I am equal parts thrilled and terrified",
    "The party was incredible but now I have to clean everything up",
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

# ---------------------------------------------------------------------------
# DEV SET — 40 hand-labeled examples, 10/class
# Used by the autoresearch loop for keep/discard decisions.
# Sources: UNLABELED_EXAMPLES (hand-labeled) + curated new examples.
# NO overlap with TEST_TEXTS.
# ---------------------------------------------------------------------------

DEV_TEXTS = [
    # -- POSITIVE (10) --
    # From UNLABELED_EXAMPLES (hand-labeled)
    "Finally finished my thesis after months of work, feels surreal",
    "Had the most amazing brunch with my best friends today \U0001f604",
    "My dog learned a new trick and I am so proud of him",
    "Got a surprise care package from my mom in the mail",
    "Baked cookies from scratch for the first time and they turned out perfect",
    "Got an A on the exam I was most worried about",
    "Random stranger complimented my outfit and made my whole week",
    # Curated new examples
    "Woke up early and watched the sunrise with a cup of coffee, pure peace",
    "My best friend flew in to surprise me for my birthday, I literally cried",
    "Finally learned to parallel park and I feel invincible",

    # -- NEGATIVE (10) --
    # From UNLABELED_EXAMPLES (hand-labeled)
    "Got ghosted after what I thought was a great first date",
    "My flight got cancelled and I missed the whole event",
    "Found out my best friend has been talking behind my back",
    "Woke up with a terrible migraine and it's not getting better",
    "My laptop crashed and I lost three hours of unsaved work",
    "Got a parking ticket right outside my own apartment",
    "Had a full blown panic attack in public today, so embarrassing",
    # Curated new examples
    "Spent an hour on hold with customer support and they hung up on me",
    "Every single thing I cooked tonight burned, I give up",
    "My roommate ate my clearly labeled leftovers again, I'm done",

    # -- NEUTRAL (10) --
    # From UNLABELED_EXAMPLES (hand-labeled)
    "The meeting ran about fifteen minutes over schedule",
    "I usually take the bus but decided to walk today",
    "I switched my phone to dark mode a few weeks ago",
    "Printed out my notes before the lecture",
    "The new coffee shop on my street opened this morning",
    "My roommate moved some furniture around this weekend",
    "Set a reminder to call the dentist tomorrow",
    # Curated new examples
    "The elevator was out of order so I took the stairs",
    "My package arrived a day earlier than estimated",
    "The vending machine was out of my usual snack so I picked a different one",

    # -- MIXED (10) --
    # From UNLABELED_EXAMPLES (hand-labeled)
    "Made it through the week, barely, but made it",
    "The movie was great but the ending ruined the whole thing",
    "Happy to be done with school but kind of scared about what's next",
    "My interview went okay I think, hard to tell honestly",
    "Finally talked to my ex, it went better than expected but still hurts",
    "Finished the race which is amazing but my knee is really sore",
    "Feeling productive but also kind of burnt out at the same time",
    # Curated new examples
    "Got a tattoo I love but the healing process is miserable",
    "Adopted a rescue dog, she's adorable but won't stop chewing everything",
    "Cleaned the whole apartment but now I'm too tired to enjoy it",
]

DEV_LABELS = [
    # Positive (10)
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    # Negative (10)
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative", "negative",
    # Neutral (10)
    "neutral", "neutral", "neutral", "neutral", "neutral",
    "neutral", "neutral", "neutral", "neutral", "neutral",
    # Mixed (10)
    "mixed", "mixed", "mixed", "mixed", "mixed",
    "mixed", "mixed", "mixed", "mixed", "mixed",
]

assert len(DEV_TEXTS) == 40, f"Expected 40 dev examples, got {len(DEV_TEXTS)}"
assert len(DEV_LABELS) == 40, f"Expected 40 dev labels, got {len(DEV_LABELS)}"

# ---------------------------------------------------------------------------
# VALIDATION SET — 40 hand-labeled examples, 10/class
# For hyperparameter selection and overfitting detection.
# NO overlap with TEST_TEXTS or DEV_TEXTS.
# ---------------------------------------------------------------------------

VAL_TEXTS = [
    # -- POSITIVE (10) --
    # From UNLABELED_EXAMPLES (hand-labeled)
    "The sunset tonight was absolutely breathtaking \U0001f305",
    "First day at the gym in months and it felt so good",
    "Just booked flights for my dream vacation, beyond excited",
    "My presentation went really well and the team loved it",
    "Finally paid off my credit card debt \U0001f973",
    "Professor posted the grades and I actually did better than I thought",
    "Study group was actually helpful today which never happens",
    # Curated new examples
    "Landed my first freelance client and they loved my portfolio",
    "Cooked a full meal from scratch for the first time and it was delicious",
    "Passed my certification exam on the first try, all those late nights paid off",

    # -- NEGATIVE (10) --
    # From UNLABELED_EXAMPLES (hand-labeled)
    "The restaurant got my order wrong and I'm still hungry \U0001f61e",
    "My cat has been sick all week and I'm really worried \U0001f622",
    "Missed my alarm and was late to the most important meeting of the month",
    "My phone screen cracked after dropping it once",
    "Pulled an all-nighter and still feel like I don't know anything",
    "Skipped my medication twice this week and really felt it",
    "Cancelled plans again and I know my friends are getting tired of it",
    # Curated new examples
    "My car broke down on the highway in the middle of nowhere",
    "Opened my bank account and realized I have less than I thought",
    "Tripped in front of the entire lecture hall and wanted to disappear",

    # -- NEUTRAL (10) --
    # From UNLABELED_EXAMPLES (hand-labeled)
    "I read about forty pages before bed",
    "Third-wheeled my friends' date night and honestly it was fine",
    "Said no to plans for the first time and didn't feel guilty",
    "Spent the whole weekend alone and somehow that was exactly what I needed",
    "Took a rest day instead of pushing through and honestly needed it",
    # Curated new examples
    "Swapped my morning coffee for tea this week just to try it",
    "The construction next door started at seven again",
    "Reorganized my bookshelf by color instead of genre",
    "Picked up my prescription on the way home from work",
    "My neighbor introduced himself after living next door for a year",

    # -- MIXED (10) --
    # From UNLABELED_EXAMPLES (hand-labeled)
    "The party was fun but I'm exhausted and overstimulated now",
    "Had a long overdue conversation with my dad and it went better than expected",
    "Been drinking more water this week and honestly feel different",
    "Got an extension so now I have no excuse not to do it right",
    "My sister and I went for a walk and talked for two hours straight",
    # Curated new examples
    "Won a scholarship but it means moving away from everyone I know",
    "The concert was incredible but I lost my voice screaming along",
    "Started therapy and it's helping but every session is emotionally exhausting",
    "Published my first blog post, proud of it but terrified of the feedback",
    "Quit my toxic job which feels freeing but now I have no income",
]

VAL_LABELS = [
    # Positive (10)
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    # Negative (10)
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative", "negative",
    # Neutral (10)
    "neutral", "neutral", "neutral", "neutral", "neutral",
    "neutral", "neutral", "neutral", "neutral", "neutral",
    # Mixed (10)
    "mixed", "mixed", "mixed", "mixed", "mixed",
    "mixed", "mixed", "mixed", "mixed", "mixed",
]

assert len(VAL_TEXTS) == 40, f"Expected 40 val examples, got {len(VAL_TEXTS)}"
assert len(VAL_LABELS) == 40, f"Expected 40 val labels, got {len(VAL_LABELS)}"

# ---------------------------------------------------------------------------
# Fast lookup sets
# ---------------------------------------------------------------------------

_TEST_TEXTS_SET = frozenset(TEST_TEXTS)
_DEV_TEXTS_SET = frozenset(DEV_TEXTS)
_VAL_TEXTS_SET = frozenset(VAL_TEXTS)
_ALL_EVAL_TEXTS_SET = _TEST_TEXTS_SET | _DEV_TEXTS_SET | _VAL_TEXTS_SET
_SEED_TEXTS_SET = frozenset(SAMPLE_POSTS)

# ---------------------------------------------------------------------------
# Integrity checksums — computed once, verified every evaluation
# ---------------------------------------------------------------------------

def _compute_checksum(texts, labels):
    data = json.dumps(
        [{"text": t, "label": l} for t, l in zip(texts, labels)],
        sort_keys=True,
    )
    return hashlib.sha256(data.encode()).hexdigest()


_EXPECTED_TEST_CHECKSUM = _compute_checksum(TEST_TEXTS, TEST_LABELS)
_EXPECTED_DEV_CHECKSUM = _compute_checksum(DEV_TEXTS, DEV_LABELS)
_EXPECTED_VAL_CHECKSUM = _compute_checksum(VAL_TEXTS, VAL_LABELS)


def _verify_integrity():
    """Verify all three datasets have not been tampered with."""
    actual_test = _compute_checksum(TEST_TEXTS, TEST_LABELS)
    assert actual_test == _EXPECTED_TEST_CHECKSUM, (
        f"Test dataset integrity check failed!\n"
        f"Expected: {_EXPECTED_TEST_CHECKSUM}\n"
        f"Got:      {actual_test}"
    )

    actual_dev = _compute_checksum(DEV_TEXTS, DEV_LABELS)
    assert actual_dev == _EXPECTED_DEV_CHECKSUM, (
        f"Dev dataset integrity check failed!\n"
        f"Expected: {_EXPECTED_DEV_CHECKSUM}\n"
        f"Got:      {actual_dev}"
    )

    actual_val = _compute_checksum(VAL_TEXTS, VAL_LABELS)
    assert actual_val == _EXPECTED_VAL_CHECKSUM, (
        f"Val dataset integrity check failed!\n"
        f"Expected: {_EXPECTED_VAL_CHECKSUM}\n"
        f"Got:      {actual_val}"
    )


def _verify_no_overlap():
    """Assert no overlap between any pair of sets, or with seed training data."""
    test_dev = _TEST_TEXTS_SET & _DEV_TEXTS_SET
    assert not test_dev, f"LEAKAGE: test/dev overlap ({len(test_dev)} examples): {test_dev}"

    test_val = _TEST_TEXTS_SET & _VAL_TEXTS_SET
    assert not test_val, f"LEAKAGE: test/val overlap ({len(test_val)} examples): {test_val}"

    dev_val = _DEV_TEXTS_SET & _VAL_TEXTS_SET
    assert not dev_val, f"LEAKAGE: dev/val overlap ({len(dev_val)} examples): {dev_val}"

    seed_test = _SEED_TEXTS_SET & _TEST_TEXTS_SET
    assert not seed_test, f"LEAKAGE: seed/test overlap: {seed_test}"

    seed_dev = _SEED_TEXTS_SET & _DEV_TEXTS_SET
    assert not seed_dev, f"LEAKAGE: seed/dev overlap: {seed_dev}"

    seed_val = _SEED_TEXTS_SET & _VAL_TEXTS_SET
    assert not seed_val, f"LEAKAGE: seed/val overlap: {seed_val}"


# ---------------------------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------------------------

def evaluate(predictions: list, labels: list) -> dict:
    """
    Compute metrics for a list of predictions against provided labels.

    Args:
        predictions: list of predicted labels (must be from CLASSES)
        labels: ground-truth labels to evaluate against

    Returns:
        dict with:
            macro_f1   - PRIMARY metric (higher is better)
            accuracy   - secondary metric
            per_class  - {class_name: {precision, recall, f1, support}}
    """
    _verify_integrity()

    assert len(predictions) == len(labels), (
        f"predictions length ({len(predictions)}) != labels length ({len(labels)})"
    )

    # Sanitize predictions: invalid labels become "mixed" (penalized naturally)
    valid = set(CLASSES)
    sanitized = [p if p in valid else "mixed" for p in predictions]

    macro_f1 = f1_score(labels, sanitized, labels=CLASSES, average="macro", zero_division=0)
    accuracy = accuracy_score(labels, sanitized)

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
# Cross-Validation
# ---------------------------------------------------------------------------

def cross_validate(build_model_fn, texts, labels, k=5):
    """
    Stratified k-fold cross-validation.

    Args:
        build_model_fn: callable() -> model with .fit(texts, labels) and .predict(texts)
        texts: list of strings
        labels: list of label strings
        k: number of folds (default 5)

    Returns:
        (mean_f1, std_f1) — macro F1 across folds
    """
    texts_arr = np.array(texts)
    labels_arr = np.array(labels)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    fold_f1s = []

    for train_idx, val_idx in skf.split(texts_arr, labels_arr):
        model = build_model_fn()
        model.fit(list(texts_arr[train_idx]), list(labels_arr[train_idx]))
        preds = model.predict(list(texts_arr[val_idx]))
        f1 = f1_score(
            list(labels_arr[val_idx]), list(preds),
            labels=CLASSES, average="macro", zero_division=0,
        )
        fold_f1s.append(f1)

    return float(np.mean(fold_f1s)), float(np.std(fold_f1s))


# ---------------------------------------------------------------------------
# McNemar's Test for Significance
# ---------------------------------------------------------------------------

def mcnemar_test(preds_a: list, preds_b: list, labels: list) -> float:
    """
    McNemar's test comparing two classifiers on the same dataset.

    Tests whether the two classifiers disagree in a systematically
    different way (one is better than the other).

    Args:
        preds_a: predictions from classifier A
        preds_b: predictions from classifier B
        labels: ground-truth labels

    Returns:
        p_value (float). p < 0.05 suggests a significant difference.
    """
    assert len(preds_a) == len(preds_b) == len(labels), (
        "All inputs must have the same length"
    )

    # Build the 2x2 contingency table:
    #   b = A correct, B wrong
    #   c = A wrong, B correct
    b = 0  # A right, B wrong
    c = 0  # A wrong, B right

    for pa, pb, true in zip(preds_a, preds_b, labels):
        a_correct = (pa == true)
        b_correct = (pb == true)
        if a_correct and not b_correct:
            b += 1
        elif not a_correct and b_correct:
            c += 1

    # Use the exact binomial test for small counts, chi-squared for large
    n = b + c
    if n == 0:
        return 1.0  # No disagreement at all

    if n < 25:
        # Exact binomial test: P(X >= max(b,c)) under H0: p=0.5
        from scipy.stats import binom_test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                p_value = binom_test(b, n, 0.5)
            except Exception:
                from scipy.stats import binomtest
                result = binomtest(b, n, 0.5)
                p_value = result.pvalue
    else:
        # Chi-squared approximation with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        from scipy.stats import chi2 as chi2_dist
        p_value = 1.0 - chi2_dist.cdf(chi2, df=1)

    return float(p_value)


# ---------------------------------------------------------------------------
# Results Printing — greppable output format
# ---------------------------------------------------------------------------

def print_results(
    dev_metrics: dict,
    val_metrics: dict,
    cv_mean: float = None,
    cv_std: float = None,
    elapsed: float = None,
) -> None:
    """
    Print results in a greppable format matching autoresearch conventions.

    Extraction: grep "^dev_macro_f1:" run.log
    """
    print("---")
    print(f"dev_macro_f1:     {dev_metrics['macro_f1']:.6f}")
    print(f"dev_accuracy:     {dev_metrics['accuracy']:.6f}")
    print(f"val_macro_f1:     {val_metrics['macro_f1']:.6f}")
    print(f"val_accuracy:     {val_metrics['accuracy']:.6f}")
    if cv_mean is not None:
        print(f"cv_mean:          {cv_mean:.6f}")
    if cv_std is not None:
        print(f"cv_std:           {cv_std:.6f}")
    for cls in CLASSES:
        dev_f1 = dev_metrics["per_class"][cls]["f1"]
        val_f1 = val_metrics["per_class"][cls]["f1"]
        print(f"dev_{cls}_f1: {dev_f1:.6f}")
        print(f"val_{cls}_f1: {val_f1:.6f}")
    if elapsed is not None:
        print(f"elapsed_seconds:  {elapsed:.1f}")


# ---------------------------------------------------------------------------
# Data Utilities
# ---------------------------------------------------------------------------

def get_seed_training_data() -> tuple:
    """
    Return the seed labeled training data from dataset.py.
    Returns (texts: list[str], labels: list[str]) -- 10 examples.
    """
    return list(SAMPLE_POSTS), list(TRUE_LABELS)


def get_unlabeled_pool() -> list:
    """
    Return unlabeled examples from dataset.py, EXCLUDING any that appear
    in dev, val, test, or seed sets. This prevents any data leakage.
    """
    return [t for t in UNLABELED_EXAMPLES if t not in _ALL_EVAL_TEXTS_SET and t not in _SEED_TEXTS_SET]


def get_dev_data() -> tuple:
    """
    Return the dev set for keep/discard decisions in the autoresearch loop.
    Returns (texts: list[str], labels: list[str]) -- 40 examples, 10/class.
    """
    return list(DEV_TEXTS), list(DEV_LABELS)


def get_val_data() -> tuple:
    """
    Return the validation set for hyperparameter selection / overfitting detection.
    Returns (texts: list[str], labels: list[str]) -- 40 examples, 10/class.
    """
    return list(VAL_TEXTS), list(VAL_LABELS)


def get_test_data() -> tuple:
    """
    Return the fixed held-out test set (final report only).
    Returns (texts: list[str], labels: list[str]) -- 60 examples, 15/class.
    """
    return list(TEST_TEXTS), list(TEST_LABELS)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("prepare_v2.py — self-test")
    print("=" * 60)

    # 1. Integrity checksums
    _verify_integrity()
    print("[PASS] Integrity checksums verified for all three sets.")

    # 2. No overlap
    _verify_no_overlap()
    print("[PASS] No overlap between any pair of sets or with seed data.")

    # 3. Sizes
    test_t, test_l = get_test_data()
    dev_t, dev_l = get_dev_data()
    val_t, val_l = get_val_data()
    seed_t, seed_l = get_seed_training_data()
    pool = get_unlabeled_pool()

    print(f"\nDataset sizes:")
    print(f"  Test set:       {len(test_t)} examples")
    print(f"  Dev set:        {len(dev_t)} examples")
    print(f"  Val set:        {len(val_t)} examples")
    print(f"  Seed training:  {len(seed_t)} examples")
    print(f"  Unlabeled pool: {len(pool)} examples (after excluding all eval + seed sets)")

    # 4. Class balance
    print(f"\nClass distributions:")
    for name, labels in [("Test", test_l), ("Dev", dev_l), ("Val", val_l)]:
        dist = Counter(labels)
        print(f"  {name:5s}: {dict(dist)}")
        for cls in CLASSES:
            assert dist[cls] > 0, f"{name} set missing class {cls}"

    # 5. Uniqueness within each set
    assert len(set(test_t)) == len(test_t), "Duplicate texts in TEST set"
    assert len(set(dev_t)) == len(dev_t), "Duplicate texts in DEV set"
    assert len(set(val_t)) == len(val_t), "Duplicate texts in VAL set"
    print("[PASS] No duplicate texts within any set.")

    # 6. SHA-256 checksums
    print(f"\nSHA-256 checksums:")
    print(f"  Test: {_EXPECTED_TEST_CHECKSUM}")
    print(f"  Dev:  {_EXPECTED_DEV_CHECKSUM}")
    print(f"  Val:  {_EXPECTED_VAL_CHECKSUM}")

    # 7. Quick smoke test of evaluate()
    dummy_preds = dev_l[:]  # perfect predictions
    metrics = evaluate(dummy_preds, dev_l)
    assert metrics["macro_f1"] == 1.0, f"Perfect preds should give F1=1.0, got {metrics['macro_f1']}"
    print("[PASS] evaluate() smoke test passed (perfect preds -> F1=1.0).")

    # 8. Quick smoke test of mcnemar_test()
    preds_same = ["positive"] * 10
    labels_same = ["positive"] * 10
    p = mcnemar_test(preds_same, preds_same, labels_same)
    assert p == 1.0, f"Identical classifiers should give p=1.0, got {p}"
    print("[PASS] mcnemar_test() smoke test passed (identical preds -> p=1.0).")

    # 9. Verify pool exclusions
    pool_set = set(pool)
    assert not (pool_set & _TEST_TEXTS_SET), "LEAKAGE: pool contains test examples"
    assert not (pool_set & _DEV_TEXTS_SET), "LEAKAGE: pool contains dev examples"
    assert not (pool_set & _VAL_TEXTS_SET), "LEAKAGE: pool contains val examples"
    assert not (pool_set & _SEED_TEXTS_SET), "LEAKAGE: pool contains seed examples"
    print("[PASS] Unlabeled pool has no overlap with any labeled set.")

    print(f"\n{'=' * 60}")
    print("All checks passed.")
    print(f"{'=' * 60}")
