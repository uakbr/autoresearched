# autoresearch v3 — SetFit Embeddings

This is autoresearch v3: sentence transformers replace bag-of-words. The model actually learns context, sarcasm, and negation scope from pre-trained embeddings instead of relying on hand-coded heuristics.

## Setup

1. **Agree on a run tag** (e.g. `mar25v3`). Branch `autoresearch/<tag>` must not exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the in-scope files**:
   - `prepare_v2.py` — fixed evaluation harness (dev/val/test splits, CV, integrity checks). Do not modify.
   - `train_v3.py` — the file you modify. SetFit model, training data, hyperparameters.
   - `backend/dataset.py` — word lists, training data, unlabeled pool. Read-only.
   - `backend/mood_analyzer.py` — rule-based classifier. Read-only.
   - `backend/active_learner.py` — clustering, uncertainty sampling. Read-only but you can import.
4. **Verify deps**: `source .venv/bin/activate && python -c "from setfit import SetFitModel; print('OK')"`
5. **Initialize results_v3.tsv** with header row.
6. **Run baseline**: `python train_v3.py > run.log 2>&1`

## What you CAN do

- Modify `train_v3.py` — this is the only file you edit.
- Change SetFit hyperparameters (num_iterations, num_epochs, batch_size, learning_rate, model name).
- Add/remove/modify training examples (EXTRA_TRAIN list).
- Add auxiliary features alongside SetFit embeddings (rb_score, custom features).
- Change the classification head (SetFit default, or extract embeddings + train your own classifier).
- Use active learning from `active_learner.py` to expand training data.
- Mutate word lists at runtime (these feed the rb_score feature if you use it).

## What you CANNOT do

- Modify `prepare_v2.py` (evaluation harness, dev/val/test sets).
- Modify any file in `backend/`.
- Install new packages.
- Add hand-coded post-prediction heuristics (the point of v3 is to LEARN, not pattern-match).

## The goal

**Maximize cv_mean** (5-fold cross-validation macro-F1). This is the honest generalization metric.

Secondary goals (in order):
1. Maximize val_macro_f1 (held-out validation)
2. Maximize dev_macro_f1 (dev set for rapid iteration)
3. Minimize cv_std (lower variance = more robust)
4. Minimize cv-dev gap (closer = less overfitting)

**NOTE**: cv_mean is HIGHER is better. Current baseline: 0.856.

## Output format

```
---
dev_macro_f1:     0.787857
val_macro_f1:     0.642965
cv_mean:          0.855760
cv_std:           0.033097
...
```

Extract: `grep "^cv_mean:" run.log`

## Logging results

`results_v3.tsv` (tab-separated, 8 columns):

```
commit	dev_macro_f1	val_macro_f1	cv_mean	cv_std	status	category	description
```

Categories: `data`, `hyperparams`, `architecture`, `features`, `words`, `active_learning`

## Decision rules

```
IF cv_mean > best_cv + 0.005:
    KEEP (clear generalization improvement)
ELIF cv_mean > best_cv - 0.003 AND val_macro_f1 improved:
    KEEP (val improvement with stable CV)
ELIF cv_mean == best_cv AND code is simpler:
    KEEP (simplification)
ELSE:
    DISCARD
```

**Overfitting check**: If dev improves but cv_mean drops, that's overfitting. DISCARD.
**Plateau detection**: After 10 consecutive discards, do a systematic grid search over num_iterations=[10,20,40], batch_size=[8,16,32], num_epochs=[1,2,3].

## The experiment loop

LOOP FOREVER:

1. Read git state and last results
2. Check which metric is weakest (per-class dev F1, val F1)
3. Design experiment targeting that weakness
4. Edit train_v3.py
5. git commit
6. `python train_v3.py > run.log 2>&1`
7. Extract metrics: `grep "^cv_mean:\|^dev_macro_f1:\|^val_macro_f1:\|^cv_std:" run.log`
8. Apply decision rules
9. Log to results_v3.tsv
10. If discard: `git reset --hard HEAD~1`
11. Continue. **NEVER STOP.**

## Meta-learning (every 10 experiments)

After every 10 experiments:
1. Count success rate per category in results_v3.tsv
2. Identify which category has highest keep rate
3. Bias next 10 experiments toward high-success categories
4. If all categories have low success, try the LEAST-explored one

## Experiment ideas (ordered by expected impact)

**Tier 1 — High leverage:**
1. Tune num_iterations (10, 20, 40, 80) — controls contrastive pair diversity
2. Tune num_epochs (1, 2, 3) — more passes over contrastive pairs
3. Tune batch_size (8, 16, 32) — affects gradient noise
4. Add more diverse training examples (especially for weakest class)
5. Try different base model: `sentence-transformers/all-mpnet-base-v2` (768-dim, higher quality)
6. Add rb_score as auxiliary feature alongside SetFit embeddings

**Tier 2 — Medium leverage:**
7. Use SetFit with `use_differentiable_head=True` (end-to-end fine-tuning)
8. Add body_learning_rate for the transformer body (default: frozen)
9. Try learning_rate for the classification head
10. Add more training examples from unlabeled pool using active learning

**Tier 3 — Experimental:**
11. Ensemble: train 3 SetFit models with different seeds, vote
12. Curriculum learning: train on easy examples first, hard examples second
13. Data augmentation: paraphrase training examples
14. Two-phase training: contrastive first, then fine-tune head on full data

## Time budget

Each experiment takes ~60-90 seconds (SetFit training on CPU). Budget: ~40-50 experiments per hour.

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The loop runs until the human interrupts you, period.
