# autoresearch v2 — Emotional-Learning

This is an experiment to have the LLM autonomously research and improve a mood classification system, using rigorous three-way evaluation splits (DEV / VAL / TEST).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar25v2`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare_v2.py` — fixed constants, evaluation harness (DEV/VAL/TEST splits), cross-validation, McNemar significance testing. Do not modify.
   - `train_v2.py` — the file you modify. ML configuration, training data, model selection.
   - `backend/dataset.py` — word lists, thresholds, training data, unlabeled pool. Read-only but you import from it and mutate globals at runtime in train_v2.py.
   - `backend/mood_analyzer.py` — rule-based classifier. Read-only but you import and use it.
   - `backend/ml_model.py` — ML classifier utilities. Read-only but you import and use it.
   - `backend/active_learner.py` — clustering, sampling, uncertainty. Read-only but you can import.
4. **Verify dependencies**: Run `source .venv/bin/activate && python -c "import sklearn; print('OK')"`. If it fails, run `pip install -r requirements.txt`.
5. **Initialize results_v2.tsv**: Create `results_v2.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on CPU (M4 Max MacBook, Apple Silicon). The training script runs fast (typically under 5 seconds for scikit-learn). You launch it simply as: `python train_v2.py`.

**What you CAN do:**
- Modify `train_v2.py` — this is the only file you edit. Everything is fair game: ML classifier choice, hyperparameters, feature engineering, training data expansion (auto-labeling from unlabeled pool), rule-based configuration, ensemble methods, preprocessing, etc.
- Import anything from `backend/` and `prepare_v2.py`.
- Use any technique available in scikit-learn, numpy, scipy.
- Mutate `dataset.py` globals at runtime from within `train_v2.py` (e.g. `add_positive_word(...)`, `set_label_thresholds(...)`).

**What you CANNOT do:**
- Modify `prepare_v2.py`. It is read-only. It contains the fixed evaluation, DEV/VAL/TEST datasets, and constants.
- Modify any file in `backend/` directly (dataset.py, mood_analyzer.py, ml_model.py, active_learner.py).
- Install new packages or add dependencies. You can only use what's already installed (scikit-learn, numpy, scipy, emoji, pandas).
- Modify the `evaluate()` function or any evaluation dataset.

**The goal is simple: get the highest dev_macro_f1.** NOTE: HIGHER is better (this is the opposite of autoresearch's val_bpb which is LOWER is better). dev_macro_f1 ranges from 0.0 (worst) to 1.0 (perfect). It is the macro-averaged F1 score across all 4 classes (positive, negative, neutral, mixed), computed by sklearn on the DEV set.

**Time budget** is a soft constraint. Each experiment should complete in under 30 seconds. If a run exceeds 60 seconds, kill it and treat it as a failure.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.01 dev_macro_f1 improvement that adds 50 lines of hacky code? Probably not worth it. A 0.01 improvement from simplifying? Definitely keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Key things to know

1. **The rule-based model cannot predict "neutral"** — it only outputs positive/negative/mixed (see mood_analyzer.py line 122-127). The test set has 15 neutral examples, the dev set has 10. This means the rule-based baseline gets 0% recall on neutral. Fixing this in train_v2.py is a high-leverage improvement.

2. **Only 10 seed training examples** — the ML model starts with very little data. The highest-leverage early experiments are expanding the training set by auto-labeling examples from the unlabeled pool (use `get_unlabeled_pool()` from prepare_v2.py).

3. **Sarcasm is a known failure mode** — examples like "Oh great, another meeting" are labeled negative in the test set but the rule-based model scores them as positive. The ML model also fails unless it sees sarcastic training examples.

4. **debug prints in mood_analyzer.py** — lines 52 and 56 print to stdout. train_v2.py suppresses this with `contextlib.redirect_stdout`. Keep this pattern when using MoodAnalyzer.

5. **Three evaluation sets** — DEV (40 examples, 10/class) drives keep/discard. VAL (40 examples, 10/class) detects overfitting. TEST (60 examples, 15/class) is held out for final reporting only.

6. **Cross-validation** — 5-fold stratified CV on training data provides a stability estimate. Large cv_std (>0.10) suggests the model is unstable.

## Output format

Once the script finishes it prints a summary like this:

```
---
dev_macro_f1:     0.650000
dev_accuracy:     0.700000
val_macro_f1:     0.620000
val_accuracy:     0.680000
cv_mean:          0.580000
cv_std:           0.045000
dev_positive_f1:  0.750000
val_positive_f1:  0.720000
dev_negative_f1:  0.700000
val_negative_f1:  0.680000
dev_neutral_f1:   0.600000
val_neutral_f1:   0.580000
dev_mixed_f1:     0.550000
val_mixed_f1:     0.520000
elapsed_seconds:  1.2
```

You can extract the key metric from the log file:

```
grep "^dev_macro_f1:" run.log
```

## Logging results

When an experiment is done, log it to `results_v2.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 7 columns:

```
commit	dev_macro_f1	val_macro_f1	cv_mean	cv_std	status	description
```

1. git commit hash (short, 7 chars)
2. dev_macro_f1 achieved (e.g. 0.650000) — use 0.000000 for crashes
3. val_macro_f1 achieved (e.g. 0.620000) — use 0.000000 for crashes
4. cv_mean — cross-validation mean macro_f1 (0.000000 for crashes)
5. cv_std — cross-validation standard deviation (0.000000 for crashes)
6. status: `keep`, `discard`, `crash`, or `overfit` (kept but flagged)
7. short text description of what this experiment tried

Example:

```
commit	dev_macro_f1	val_macro_f1	cv_mean	cv_std	status	description
a1b2c3d	0.580000	0.560000	0.520000	0.045000	keep	baseline
b2c3d4e	0.650000	0.630000	0.580000	0.040000	keep	switch to SVM C=5.0
c3d4e5f	0.640000	0.500000	0.550000	0.060000	overfit	added 50 features (dev up but val dropped >0.03)
d4e5f6g	0.000000	0.000000	0.000000	0.000000	crash	import error from bad refactor
e5f6g7h	0.680000	0.660000	0.600000	0.035000	keep	auto-label 30 unlabeled examples at confidence>0.7
```

## The decision rule

When deciding whether to keep or discard a change, use these rules in order:

1. **Clear improvement**: `dev_macro_f1 > best + 0.005` → **KEEP**
2. **Balanced trade**: `dev_macro_f1 > best - 0.002` AND per-class balance improved (the weakest class F1 on dev went up) → **KEEP**
3. **Simplification**: `dev_macro_f1 == best` (within 0.001) AND the code is simpler (fewer lines, fewer features, removed hacks) → **KEEP**
4. **Otherwise** → **DISCARD**

**Overfitting flag**: If dev_macro_f1 improves (rule 1 or 2 triggers KEEP) but val_macro_f1 drops by more than 0.03 compared to the previous best val_macro_f1, still keep the change but flag it as `overfit` in the status column. Note in the description: "OVERFIT WARNING: dev improved but val dropped by X.XX". This is a yellow flag — if two consecutive experiments get flagged, consider reverting to the last clean keep.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar25v2`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. **Pre-experiment analysis**: grep the last run.log for per-class F1 scores. Identify the weakest class on the dev set. Target that class in the next experiment.
3. Tune `train_v2.py` with an experimental idea by directly hacking the code.
4. git commit.
5. Run the experiment: `python train_v2.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context).
6. Read out the results: `grep "^dev_macro_f1:\|^val_macro_f1:\|^cv_mean:\|^cv_std:" run.log`
7. Also check per-class: `grep "_f1:" run.log` to identify weakest classes for next iteration.
8. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
9. Record the results in the tsv (NOTE: do not commit the results_v2.tsv file, leave it untracked by git).
10. Apply the decision rule (see above) to determine keep/discard/overfit.
11. If keeping, you "advance" the branch, keeping the git commit.
12. If discarding, you git reset back to where you started.
13. **Plateau detection**: If 10 consecutive experiments are discarded, switch strategy to systematic grid search. Try combinations of: C values [0.1, 0.5, 1.0, 5.0, 10.0, 50.0], max_features [100, 200, 300, 500], ngram_range [(1,1), (1,2), (1,3)]. Log each grid point. After grid search, resume creative experimentation with the best configuration found.
14. Continue to next experiment.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take under 30 seconds. If a run exceeds 60 seconds, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the backend files for new angles, try combining previous near-misses, try more radical approaches. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. Each experiment takes you under 1 minute total, so you can run approximately 60 per hour, for a total of about 480 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Experiment ideas to get you started

Here are categories of experiments roughly ordered from highest to lowest leverage:

### Tier 1 — High leverage (try first)
1. **Auto-labeling**: Use ML model's predict_proba to label unlabeled pool examples with high confidence, expanding training data
2. **Self-training**: Iterative bootstrap — train, high-confidence auto-label, retrain, repeat
3. **Sentence transformers**: If sentence-transformers is installed, use pre-trained embeddings as features (check availability first with try/except)
4. **Active learning**: Use uncertainty sampling from active_learner.py to find the most informative unlabeled examples, hand-label the patterns, add to EXTRA_TRAIN
5. **Learning curves**: Plot (or compute) learning curves to determine if more data or more features would help most

### Tier 2 — Medium leverage
6. **ML classifier swap**: Try SVM variants (different kernels), GradientBoosting, RandomForest, MultinomialNB, etc.
7. **Ensemble**: Combine rule-based + ML predictions via voting or stacking
8. **Feature engineering**: Add features beyond the current 15 (POS tag patterns, character ngrams, punctuation density ratios)
9. **TF-IDF tuning**: Change max_features, ngram_range, min_df, max_df, sublinear_tf, analyzer='char_wb'
10. **Cascade threshold tuning**: Adjust the neutral/emotional boundary in stage 1 using decision_function scores

### Tier 3 — Lower leverage (refinement)
11. **Word list expansion**: Add sentiment words the current lists are missing (check dev errors for clues)
12. **Threshold tuning**: Adjust positive_above/negative_below thresholds
13. **Sarcasm detection**: More sophisticated sarcasm patterns (positive words + negative context)
14. **Data augmentation**: Generate synthetic training examples via templates or synonym substitution
15. **Slang handling**: Handle "no cap", "hit different", "fr" as phrases not individual tokens
16. **Class-specific calibration**: Different confidence thresholds per class for the cascade
