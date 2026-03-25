#!/usr/bin/env python3
"""
Emotional-Learning experiment script (v3 — SetFit baseline).

Uses SetFit (few-shot fine-tuning of sentence-transformers) with
all-MiniLM-L6-v2 as the base model. No hand-coded heuristics —
the embedding model handles context directly.

Evaluation uses the same prepare_v2.py harness as v2
(DEV / VAL splits, cross-validation, print_results).

Usage: python train_v3.py
"""

import sys
import os
import time
import io
import contextlib
import random
import warnings

import numpy as np

# Backend imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from prepare_v2 import (
    TIME_BUDGET, RANDOM_SEED, CLASSES,
    evaluate, print_results,
    get_seed_training_data,
    get_dev_data, get_val_data, get_test_data,
    cross_validate,
)

from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset

# Reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

t_start = time.time()

# ============================================================
# LABEL MAPPING
# ============================================================

label2id = {cls: i for i, cls in enumerate(CLASSES)}
id2label = {i: cls for i, cls in enumerate(CLASSES)}

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

print(f"Training data: {len(train_texts)} examples")
print(f"  Class distribution: { {cls: train_labels.count(cls) for cls in CLASSES} }")

# ============================================================
# SETFIT MODEL TRAINING
# ============================================================

# Convert string labels to integers
train_label_ids = [label2id[l] for l in train_labels]

train_dataset = Dataset.from_dict({
    "text": train_texts,
    "label": train_label_ids,
})

print("\nLoading SetFit model (all-mpnet-base-v2)...")
model = SetFitModel.from_pretrained(
    "sentence-transformers/all-mpnet-base-v2",
    labels=CLASSES,
)

args = TrainingArguments(
    batch_size=4,
    num_epochs=1,
    num_iterations=10,  # number of text pairs for contrastive learning
    seed=RANDOM_SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
)

print("Training SetFit model...")
trainer.train()
print("Training complete.\n")


# ============================================================
# PREDICTION HELPER
# ============================================================

def _decode_predictions(preds):
    """Convert SetFit predictions to string labels.

    SetFit may return integer IDs or string labels depending on
    whether the model was loaded with labels=CLASSES.  Handle both.
    """
    if hasattr(preds, "tolist"):
        preds = preds.tolist()
    result = []
    for p in preds:
        if isinstance(p, str):
            result.append(p)
        else:
            result.append(id2label[int(p)])
    return result


def predict_setfit(texts):
    """Predict labels for a list of texts using the trained SetFit model."""
    preds = model.predict(texts)
    return _decode_predictions(preds)


# ============================================================
# CROSS-VALIDATION (on training data)
# ============================================================

class SetFitCV:
    """Wrapper for cross_validate: .fit(texts, labels) and .predict(texts)."""
    def __init__(self):
        self._model = None

    def fit(self, texts, labels):
        label_ids = [label2id[l] for l in labels]
        ds = Dataset.from_dict({"text": texts, "label": label_ids})

        self._model = SetFitModel.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2",
            labels=CLASSES,
        )
        cv_args = TrainingArguments(
            batch_size=4,
            num_epochs=1,
            num_iterations=10,
            seed=RANDOM_SEED,
        )
        cv_trainer = Trainer(
            model=self._model,
            args=cv_args,
            train_dataset=ds,
        )
        # Suppress SetFit progress output during CV folds
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            cv_trainer.train()

    def predict(self, texts):
        preds = self._model.predict(texts)
        return _decode_predictions(preds)


print("Running 5-fold cross-validation (this may take a minute)...")
# Suppress verbose output from SetFit during CV
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cv_mean, cv_std = cross_validate(
        build_model_fn=SetFitCV,
        texts=train_texts,
        labels=train_labels,
        k=5,
    )
print(f"CV complete: mean={cv_mean:.4f}, std={cv_std:.4f}\n")

# ============================================================
# EVALUATION
# ============================================================

# DEV set — used for keep/discard decisions
dev_texts, dev_labels = get_dev_data()
dev_preds = predict_setfit(dev_texts)
dev_metrics = evaluate(dev_preds, dev_labels)

# VAL set — used for overfitting detection
val_texts, val_labels = get_val_data()
val_preds = predict_setfit(val_texts)
val_metrics = evaluate(val_preds, val_labels)

elapsed = time.time() - t_start

# Print results in greppable format
print_results(dev_metrics, val_metrics, cv_mean, cv_std, elapsed)
