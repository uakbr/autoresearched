#!/usr/bin/env python3
"""Generate all charts for the learnings blog post."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Parse results.tsv ─────────────────────────────────────────────
rows = []
with open(os.path.join(OUT, "results.tsv")) as f:
    header = f.readline()
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 6:
            rows.append({
                'commit': parts[0],
                'macro_f1': float(parts[1]),
                'accuracy': float(parts[2]) if parts[2] != '0.000000' else None,
                'rb_macro_f1': float(parts[3]) if parts[3] != '0.000000' else None,
                'status': parts[4],
                'desc': parts[5],
            })

# ── Chart 1: macro_f1 progression (all experiments) ──────────────
plt.figure(figsize=(16, 6))
exp_nums = list(range(1, len(rows)+1))
f1_scores = [r['macro_f1'] for r in rows]
statuses = [r['status'] for r in rows]
colors = ['#2ecc71' if s == 'keep' else '#e74c3c' if f < 0.15 else '#95a5a6' for s, f in zip(statuses, f1_scores)]

plt.scatter(exp_nums, f1_scores, c=colors, s=18, alpha=0.7, zorder=2)

# Connect kept experiments with a line
kept_x = [i+1 for i, r in enumerate(rows) if r['status'] == 'keep']
kept_y = [r['macro_f1'] for r in rows if r['status'] == 'keep']
plt.plot(kept_x, kept_y, color='#2ecc71', linewidth=2.5, alpha=0.8, zorder=3, label='Kept (frontier)')

# Phase annotations
phases = [
    (1, 10, 'Phase 1\nData+Classifier', 0.02),
    (10, 15, 'Phase 2\nFeatures', 0.02),
    (15, 27, 'Phase 3\nWord Lists', 0.02),
    (27, 72, 'Phase 4\nPlateau', 0.02),
    (72, 106, 'Phase 5\nTuning', 0.02),
    (106, 112, 'Phase 6\nCascade', 0.02),
    (112, 161, 'Phase 7\nPerfection', 0.02),
]
phase_colors = ['#3498db', '#9b59b6', '#e67e22', '#e74c3c', '#1abc9c', '#f39c12', '#2ecc71']
for i, (start, end, label, _) in enumerate(phases):
    plt.axvspan(start, end, alpha=0.08, color=phase_colors[i])
    mid = (start + end) / 2
    plt.text(mid, 1.04, label, ha='center', va='bottom', fontsize=7, fontweight='bold', color=phase_colors[i])

plt.axhline(y=1.0, color='gold', linewidth=1.5, linestyle='--', alpha=0.5, label='Perfect score')
plt.xlabel('Experiment Number', fontsize=12)
plt.ylabel('macro_f1', fontsize=12)
plt.title('161 Autonomous Experiments: From 0.157 to Perfect 1.000', fontsize=14, fontweight='bold')
plt.ylim(-0.02, 1.12)
plt.xlim(0, 165)
green_patch = mpatches.Patch(color='#2ecc71', label='Kept (19%)')
gray_patch = mpatches.Patch(color='#95a5a6', label='Discarded (81%)')
plt.legend(handles=[green_patch, gray_patch], loc='lower right', fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'chart_progression.png'), dpi=150)
plt.close()
print("chart_progression.png")

# ── Chart 2: Kept experiments staircase ──────────────────────────
plt.figure(figsize=(14, 6))
kept_descs = [r['desc'][:45] for r in rows if r['status'] == 'keep']
kept_f1 = [r['macro_f1'] for r in rows if r['status'] == 'keep']
deltas = [kept_f1[0]] + [kept_f1[i] - kept_f1[i-1] for i in range(1, len(kept_f1))]

bar_colors = []
for d in deltas:
    if d > 0.05: bar_colors.append('#27ae60')
    elif d > 0.01: bar_colors.append('#2ecc71')
    elif d > 0: bar_colors.append('#82e0aa')
    else: bar_colors.append('#f0b27a')

bars = plt.barh(range(len(kept_f1)), kept_f1, color=bar_colors, edgecolor='white', height=0.7)
for i, (f1, desc) in enumerate(zip(kept_f1, kept_descs)):
    plt.text(f1 + 0.01, i, f'{f1:.3f}', va='center', fontsize=8, fontweight='bold')
plt.yticks(range(len(kept_f1)), kept_descs, fontsize=7)
plt.xlabel('macro_f1', fontsize=12)
plt.title('The 30 Kept Experiments: Each Step Toward Perfection', fontsize=13, fontweight='bold')
plt.xlim(0, 1.12)
plt.axvline(x=1.0, color='gold', linewidth=1.5, linestyle='--', alpha=0.5)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'chart_staircase.png'), dpi=150)
plt.close()
print("chart_staircase.png")

# ── Chart 3: Per-class F1 evolution ──────────────────────────────
# Manually tracked per-class data at key milestones
milestones = {
    'Exp': [0, 3, 10, 13, 22, 26, 72, 89, 106, 126, 132, 161],
    'Label': ['Baseline', 'Data', 'Features', 'C=5', 'Words', '+Words', '300feat', 'BinCV', 'Cascade', 'Sarcasm', 'Amplify', 'PERFECT'],
    'positive': [0.303, 0.369, 0.500, 0.500, 0.667, 0.846, 0.769, 0.769, 0.815, 0.929, 1.000, 1.000],
    'negative': [0.091, 0.369, 0.467, 0.467, 0.619, 0.714, 0.789, 0.800, 0.774, 0.903, 0.967, 1.000],
    'neutral':  [0.235, 0.369, 0.000, 0.000, 0.718, 0.720, 0.759, 0.722, 0.765, 0.938, 1.000, 1.000],
    'mixed':    [0.000, 0.369, 0.340, 0.340, 0.889, 0.889, 0.889, 0.929, 0.929, 0.929, 0.967, 1.000],
}

fig, ax = plt.subplots(figsize=(14, 6))
x = range(len(milestones['Exp']))
ax.plot(x, milestones['positive'], 'o-', color='#2ecc71', linewidth=2, markersize=6, label='Positive')
ax.plot(x, milestones['negative'], 's-', color='#e74c3c', linewidth=2, markersize=6, label='Negative')
ax.plot(x, milestones['neutral'], '^-', color='#3498db', linewidth=2, markersize=6, label='Neutral')
ax.plot(x, milestones['mixed'], 'D-', color='#f39c12', linewidth=2, markersize=6, label='Mixed')
ax.axhline(y=1.0, color='gold', linewidth=1, linestyle='--', alpha=0.4)
ax.set_xticks(x)
ax.set_xticklabels(milestones['Label'], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Per-Class F1 Evolution Across Key Milestones', fontsize=13, fontweight='bold')
ax.set_ylim(-0.05, 1.1)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'chart_perclass.png'), dpi=150)
plt.close()
print("chart_perclass.png")

# ── Chart 4: Experiment categories pie chart ─────────────────────
categories = {
    'Training data': 0, 'Word lists': 0, 'Features': 0,
    'Classifier/C': 0, 'Vectorizer': 0, 'Ensemble': 0,
    'Architecture': 0, 'Other': 0
}
for r in rows:
    d = r['desc'].lower()
    if any(w in d for w in ['example', 'training', 'curated', 'sarcasm', 'neutral', 'positive', 'negative', 'mixed', 'balanced', 'unlabeled', 'reconnect', 'mundane', 'achievement', 'complaint', 'aha', 'birthday', 'pessimism', 'routine']):
        categories['Training data'] += 1
    elif any(w in d for w in ['word list', 'expand word', 'add word', 'selective word', 'negative word', 'positive word', 'achievement word', 'afford', 'raising', 'weirdly', 'amplifier', 'genuinely']):
        categories['Word lists'] += 1
    elif any(w in d for w in ['feature', 'emoji', 'sentiment', 'is_short', 'avg_word', 'sarcasm start', 'pronoun', 'punctuation', 'diversity', 'comma', 'contrast', 'interaction', 'polynomial', 'rb_', 'non-linear', 'event', 'negation-a']):
        categories['Features'] += 1
    elif any(w in d for w in ['linearsvc', 'logistic', 'svm', 'svc', 'gradient', 'random_state', 'randomforest', 'sgd', 'calibrat', 'naiveb', 'multinomial', 'class_weight', 'c=', 'hinge', 'l1 penalty', 'tol=', 'dual=', 'intercept']):
        categories['Classifier/C'] += 1
    elif any(w in d for w in ['max_features', 'ngram', 'stop_word', 'countvec', 'tfidf', 'binary', 'min_df', 'max_df', 'use_idf', 'norm=', 'sublinear', 'token_pattern', 'unigram', 'char ngram', 'vectorizer', 'lowercase']):
        categories['Vectorizer'] += 1
    elif any(w in d for w in ['ensemble', 'voting', 'stacking', 'hybrid']):
        categories['Ensemble'] += 1
    elif any(w in d for w in ['stage', 'cascade', 'two-stage', 'three-stage', 'threshold', 'signal weight', 'narrow']):
        categories['Architecture'] += 1
    else:
        categories['Other'] += 1

# Remove zero categories
categories = {k: v for k, v in categories.items() if v > 0}
fig, ax = plt.subplots(figsize=(8, 8))
colors_pie = ['#2ecc71', '#e67e22', '#3498db', '#e74c3c', '#9b59b6', '#1abc9c', '#f39c12', '#95a5a6']
wedges, texts, autotexts = ax.pie(
    categories.values(), labels=categories.keys(), autopct='%1.0f%%',
    colors=colors_pie[:len(categories)], startangle=90, textprops={'fontsize': 10}
)
for t in autotexts:
    t.set_fontsize(9)
    t.set_fontweight('bold')
ax.set_title('161 Experiments by Category', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'chart_categories.png'), dpi=150)
plt.close()
print("chart_categories.png")

# ── Chart 5: Impact analysis (delta per kept experiment) ─────────
plt.figure(figsize=(12, 6))
deltas_clean = []
labels_clean = []
for i, (d, desc) in enumerate(zip(deltas, kept_descs)):
    if i == 0:
        continue  # skip baseline
    deltas_clean.append(d)
    labels_clean.append(desc[:40])

bar_c = ['#27ae60' if d > 0.03 else '#2ecc71' if d > 0.01 else '#82e0aa' if d > 0 else '#f0b27a' for d in deltas_clean]
plt.barh(range(len(deltas_clean)), deltas_clean, color=bar_c, edgecolor='white', height=0.7)
for i, d in enumerate(deltas_clean):
    plt.text(d + 0.002 if d >= 0 else d - 0.002, i, f'+{d:.3f}' if d > 0 else f'{d:.3f}', va='center', fontsize=8, ha='left' if d >= 0 else 'right')
plt.yticks(range(len(deltas_clean)), labels_clean, fontsize=7)
plt.xlabel('Delta macro_f1', fontsize=12)
plt.title('Impact of Each Kept Experiment (Delta macro_f1)', fontsize=13, fontweight='bold')
plt.axvline(x=0, color='black', linewidth=0.5)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'chart_deltas.png'), dpi=150)
plt.close()
print("chart_deltas.png")

# ── Chart 6: Keep vs Discard rate over time ──────────────────────
plt.figure(figsize=(14, 4))
window = 10
keep_rate = []
for i in range(len(rows)):
    start = max(0, i - window + 1)
    chunk = rows[start:i+1]
    rate = sum(1 for r in chunk if r['status'] == 'keep') / len(chunk)
    keep_rate.append(rate)

plt.fill_between(exp_nums, keep_rate, alpha=0.3, color='#2ecc71')
plt.plot(exp_nums, keep_rate, color='#27ae60', linewidth=1.5)
plt.axhline(y=0.19, color='#e74c3c', linewidth=1, linestyle='--', alpha=0.5, label='Overall avg (19%)')
plt.xlabel('Experiment Number', fontsize=12)
plt.ylabel(f'Keep Rate ({window}-exp rolling)', fontsize=12)
plt.title('Discovery Rate Over Time: When Did We Find Improvements?', fontsize=13, fontweight='bold')
plt.ylim(0, 1.0)
plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'chart_keeprate.png'), dpi=150)
plt.close()
print("chart_keeprate.png")

# ── Chart 7: Architecture diagram (text-based) ──────────────────
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.set_axis_off()
ax.set_title('Final Architecture: Two-Stage Cascade Classifier', fontsize=14, fontweight='bold', pad=20)

# Input
ax.add_patch(plt.Rectangle((0.5, 8.5), 3, 1, fill=True, facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=2, zorder=2))
ax.text(2, 9.0, 'Input Text', ha='center', va='center', fontsize=11, fontweight='bold')

# Feature extraction
ax.annotate('', xy=(5, 9.0), xytext=(3.5, 9.0), arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))
ax.add_patch(plt.Rectangle((5, 8.0), 4, 2, fill=True, facecolor='#d5f5e3', edgecolor='#27ae60', linewidth=2, zorder=2))
ax.text(7, 9.3, 'Feature Extraction', ha='center', va='center', fontsize=10, fontweight='bold', color='#27ae60')
ax.text(7, 8.7, 'Binary CountVec (300 bigrams)', ha='center', va='center', fontsize=8)
ax.text(7, 8.3, '+ 15 Custom Features (rb_score, etc.)', ha='center', va='center', fontsize=8)

# Stage 1
ax.annotate('', xy=(7, 7.5), xytext=(7, 8.0), arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))
ax.add_patch(plt.Rectangle((4.5, 6.0), 5, 1.5, fill=True, facecolor='#d6eaf8', edgecolor='#2980b9', linewidth=2, zorder=2))
ax.text(7, 7.1, 'Stage 1: Neutral Detector', ha='center', va='center', fontsize=10, fontweight='bold', color='#2980b9')
ax.text(7, 6.5, 'LinearSVC(C=5.0) → Neutral vs Emotional', ha='center', va='center', fontsize=9)

# Split
ax.annotate('Neutral', xy=(3, 5.0), xytext=(5.5, 6.0), arrowprops=dict(arrowstyle='->', lw=2, color='#3498db'))
ax.annotate('Emotional', xy=(10, 5.0), xytext=(8.5, 6.0), arrowprops=dict(arrowstyle='->', lw=2, color='#e67e22'))

# Neutral output
ax.add_patch(plt.Rectangle((1.5, 4.0), 3, 1, fill=True, facecolor='#aed6f1', edgecolor='#2980b9', linewidth=2, zorder=2))
ax.text(3, 4.5, '"neutral"', ha='center', va='center', fontsize=11, fontweight='bold', color='#2980b9')

# Stage 2
ax.add_patch(plt.Rectangle((8, 4.0), 5, 1.5, fill=True, facecolor='#fdebd0', edgecolor='#e67e22', linewidth=2, zorder=2))
ax.text(10.5, 5.0, 'Stage 2: Emotion Classifier', ha='center', va='center', fontsize=10, fontweight='bold', color='#e67e22')
ax.text(10.5, 4.4, 'LinearSVC(C=5.0) → Pos/Neg/Mixed', ha='center', va='center', fontsize=9)

# Stage 2 outputs
for i, (label, color) in enumerate([('positive', '#2ecc71'), ('negative', '#e74c3c'), ('mixed', '#f39c12')]):
    x = 8.5 + i * 2
    ax.annotate('', xy=(x, 3.0), xytext=(10.5, 4.0), arrowprops=dict(arrowstyle='->', lw=1.5, color=color))
    ax.add_patch(plt.Rectangle((x-0.8, 2.2), 1.6, 0.8, fill=True, facecolor=color, edgecolor='white', linewidth=2, alpha=0.3, zorder=2))
    ax.text(x, 2.6, f'"{label}"', ha='center', va='center', fontsize=9, fontweight='bold', color=color)

# Feature list
ax.text(0.3, 1.5, '15 Custom Features:', fontsize=9, fontweight='bold')
feats_left = 'emoji_count, has_but, word_count, has_question, has_exclamation, rb_score, pos_count, neg_count'
feats_right = 'neg_word_present, amp_count, sentiment_balance, emoji_sentiment, is_short, avg_word_len, has_sarcasm_start'
ax.text(0.3, 1.0, feats_left, fontsize=7, family='monospace')
ax.text(0.3, 0.6, feats_right, fontsize=7, family='monospace')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'chart_architecture.png'), dpi=150)
plt.close()
print("chart_architecture.png")

print("\nAll 7 charts generated!")
