# autoresearch — Emotional-Learning

This is an experiment to have the LLM autonomously research and improve a mood classification system.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar24`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare.py` — fixed constants, evaluation harness, test dataset. Do not modify.
   - `train.py` — the file you modify. ML configuration, training data, model selection.
   - `backend/dataset.py` — word lists, thresholds, training data, unlabeled pool. Read-only but you import from it and mutate globals at runtime in train.py.
   - `backend/mood_analyzer.py` — rule-based classifier. Read-only but you import and use it.
   - `backend/ml_model.py` — ML classifier utilities. Read-only but you import and use it.
   - `backend/active_learner.py` — clustering, sampling, uncertainty. Read-only but you can import.
4. **Verify dependencies**: Run `source .venv/bin/activate && python -c "import sklearn; print('OK')"`. If it fails, run `pip install -r requirements.txt`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on CPU (M4 Max MacBook, Apple Silicon). The training script runs fast (typically under 5 seconds for scikit-learn). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: ML classifier choice, hyperparameters, feature engineering, training data expansion (auto-labeling from unlabeled pool), rule-based configuration, ensemble methods, preprocessing, etc.
- Import anything from `backend/` and `prepare.py`.
- Use any technique available in scikit-learn, numpy, scipy.
- Mutate `dataset.py` globals at runtime from within `train.py` (e.g. `add_positive_word(...)`, `set_label_thresholds(...)`).

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, test dataset, and constants.
- Modify any file in `backend/` directly (dataset.py, mood_analyzer.py, ml_model.py, active_learner.py).
- Install new packages or add dependencies. You can only use what's already installed (scikit-learn, numpy, scipy, emoji, pandas).
- Modify the `evaluate()` function or the test dataset.

**The goal is simple: get the highest macro_f1.** NOTE: HIGHER is better (this is the opposite of autoresearch's val_bpb which is LOWER is better). macro_f1 ranges from 0.0 (worst) to 1.0 (perfect). It is the macro-averaged F1 score across all 4 classes (positive, negative, neutral, mixed), computed by sklearn.

**Time budget** is a soft constraint. Each experiment should complete in under 30 seconds. If a run exceeds 60 seconds, kill it and treat it as a failure.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.01 macro_f1 improvement that adds 50 lines of hacky code? Probably not worth it. A 0.01 improvement from simplifying? Definitely keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Key things to know

1. **The rule-based model cannot predict "neutral"** — it only outputs positive/negative/mixed (see mood_analyzer.py line 122-127). The test set has 15 neutral examples. This means the rule-based baseline gets 0% recall on neutral. Fixing this in train.py is a high-leverage improvement.

2. **Only 10 seed training examples** — the ML model starts with very little data. The highest-leverage early experiments are expanding the training set by auto-labeling examples from the unlabeled pool (use `get_unlabeled_pool()` from prepare.py).

3. **Sarcasm is a known failure mode** — examples like "Oh great, another meeting" are labeled negative in the test set but the rule-based model scores them as positive. The ML model also fails unless it sees sarcastic training examples.

4. **debug prints in mood_analyzer.py** — lines 52 and 56 print to stdout. train.py suppresses this with `contextlib.redirect_stdout`. Keep this pattern when using MoodAnalyzer.

## Output format

Once the script finishes it prints a summary like this:

```
---
macro_f1:         0.310000
accuracy:         0.420000
rb_macro_f1:      0.250000
rb_accuracy:      0.350000
positive_f1:      0.500000
negative_f1:      0.400000
neutral_f1:       0.000000
mixed_f1:         0.340000
elapsed_seconds:  0.8
```

You can extract the key metric from the log file:

```
grep "^macro_f1:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	macro_f1	accuracy	rb_macro_f1	status	description
```

1. git commit hash (short, 7 chars)
2. macro_f1 achieved (e.g. 0.420000) — use 0.000000 for crashes
3. accuracy achieved (e.g. 0.500000) — use 0.000000 for crashes
4. rb_macro_f1 — rule-based macro_f1 (tracked separately, 0.000000 for crashes)
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	macro_f1	accuracy	rb_macro_f1	status	description
a1b2c3d	0.310000	0.420000	0.250000	keep	baseline
b2c3d4e	0.420000	0.500000	0.250000	keep	switch to SVM classifier
c3d4e5f	0.380000	0.460000	0.280000	discard	random forest overfits on small data
d4e5f6g	0.000000	0.000000	0.000000	crash	import error from bad refactor
e5f6g7h	0.520000	0.580000	0.310000	keep	auto-label 30 unlabeled examples at confidence>0.7
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar24`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^macro_f1:\|^accuracy:\|^rb_macro_f1:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If macro_f1 improved (HIGHER — remember, higher is better!), you "advance" the branch, keeping the git commit
9. If macro_f1 is equal or worse, you git reset back to where you started
10. Continue to next experiment

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take under 30 seconds. If a run exceeds 60 seconds, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the backend files for new angles, try combining previous near-misses, try more radical approaches. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. Each experiment takes you under 1 minute total, so you can run approximately 60 per hour, for a total of about 480 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Experiment ideas to get you started

Here are categories of experiments roughly ordered from highest to lowest leverage:

1. **Auto-labeling**: Use ML model's predict_proba to label unlabeled pool examples with high confidence, expanding training data
2. **Self-training**: Iterative bootstrap — train → high-confidence auto-label → retrain → repeat
3. **Neutral class fix**: The rule-based model never predicts "neutral". Add logic in train.py to detect neutral scores (e.g., score 46-54 → neutral)
4. **ML classifier swap**: Try SVM, GradientBoosting, RandomForest, MultinomialNB, etc.
5. **Ensemble**: Combine rule-based + ML predictions via voting or stacking
6. **TF-IDF tuning**: Change max_features, ngram_range, min_df, max_df, sublinear_tf
7. **Word list expansion**: Add sentiment words the current lists are missing
8. **Threshold tuning**: Adjust positive_above/negative_below thresholds
9. **Feature engineering**: Add non-text features (sentence length, emoji count, punctuation density) alongside TF-IDF
10. **Sarcasm detection**: Pattern-match positive words + negative emojis as sarcastic
11. **Slang handling**: Handle "no cap", "hit different", "fr" as phrases not individual tokens
12. **Data augmentation**: Generate synthetic training examples via templates or synonym substitution
