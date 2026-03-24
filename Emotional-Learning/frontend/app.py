import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from datetime import datetime

import dataset as ds
from dataset import (
    WordSignal,
    add_sample_post,
    set_signal_weights,
    set_label_thresholds,
    UNLABELED_EXAMPLES,
)
from mood_analyzer import MoodAnalyzer
from ml_model import train_ml_model, predict_single_text
from active_learner import (
    vectorize_texts,
    cluster_texts,
    compute_elbow_data,
    sample_from_clusters,
    train_with_split,
    uncertainty_sampling,
)


@st.cache_data
def _cached_vectorize(texts_tuple):
    """Cache TF-IDF vectorization — recomputes only when texts change."""
    return vectorize_texts(list(texts_tuple))


@st.cache_data
def _cached_elbow(texts_tuple, max_k):
    """Cache elbow curve computation — expensive, should not rerun on every widget interaction."""
    _, X = vectorize_texts(list(texts_tuple))
    return compute_elbow_data(X, max_k)


st.set_page_config(page_title="Mood Machine", layout="wide")
st.title("Mood Machine Lab")

# ── Session state init ──────────────────────────────────────────────────────
if "test_posts" not in st.session_state:
    st.session_state.test_posts = []
if "test_labels" not in st.session_state:
    st.session_state.test_labels = []
if "ml_vectorizer" not in st.session_state:
    st.session_state.ml_vectorizer = None
if "ml_model" not in st.session_state:
    st.session_state.ml_model = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "ml_model_trained_at" not in st.session_state:
    st.session_state.ml_model_trained_at = None
if "rule_updated_at" not in st.session_state:
    st.session_state.rule_updated_at = None
if "training_page" not in st.session_state:
    st.session_state.training_page = 1

# Active Learning state
if "al_step" not in st.session_state:
    st.session_state.al_step = 0
if "al_raw_texts" not in st.session_state:
    st.session_state.al_raw_texts = []
if "al_X" not in st.session_state:
    st.session_state.al_X = None
if "al_cluster_labels" not in st.session_state:
    st.session_state.al_cluster_labels = None
if "al_algorithm" not in st.session_state:
    st.session_state.al_algorithm = "kmeans"
if "al_elbow_data" not in st.session_state:
    st.session_state.al_elbow_data = None
if "al_sample_indices" not in st.session_state:
    st.session_state.al_sample_indices = None
if "al_pending_labels" not in st.session_state:
    st.session_state.al_pending_labels = {}
if "al_committed_indices" not in st.session_state:
    st.session_state.al_committed_indices = set()
if "al_retrain_results" not in st.session_state:
    st.session_state.al_retrain_results = None
if "al_original_count" not in st.session_state:
    st.session_state.al_original_count = len(ds.SAMPLE_POSTS)
if "al_round" not in st.session_state:
    st.session_state.al_round = 1
if "al_accuracy_before" not in st.session_state:
    st.session_state.al_accuracy_before = None
if "al_mode" not in st.session_state:
    # "diversity" for round 0 (clustering), "uncertainty" for round 1+
    st.session_state.al_mode = "diversity"
if "al_uncertainty_indices" not in st.session_state:
    # Indices into al_raw_texts selected by uncertainty sampling
    st.session_state.al_uncertainty_indices = None
if "al_baseline_rule_acc" not in st.session_state:
    # Rule-based accuracy measured after cold start round
    st.session_state.al_baseline_rule_acc = None
if "al_baseline_ml_acc" not in st.session_state:
    # ML accuracy measured after cold start round
    st.session_state.al_baseline_ml_acc = None

# ── Auto-train ML model on first load ───────────────────────────────────────
if st.session_state.ml_model is None and len(ds.SAMPLE_POSTS) >= 4:
    _vec, _model = train_ml_model(ds.SAMPLE_POSTS, ds.TRUE_LABELS)
    st.session_state.ml_vectorizer = _vec
    st.session_state.ml_model = _model
    st.session_state.ml_model_trained_at = datetime.now()

tab1, tab2, tab3, tab4 = st.tabs(["Training Data", "Word Weights", "Testing & Evaluation", "Active Learning"])


# ── Score walkthrough helper ─────────────────────────────────────────────────
def build_walkthrough(sentence: str):
    import string as _string, re as _re, emoji as _emoji_lib

    padded = "".join(f" {ch} " if _emoji_lib.is_emoji(ch) else ch for ch in sentence)
    cleaned = "".join(ch for ch in padded if ch not in _string.punctuation).strip().lower()
    tokens = _re.sub(r"(.)\1{2,}", r"\1\1", cleaned).split()

    score = 50.0
    amplifier = 1.0
    negated = False
    steps, pending = [], []

    for token in tokens:
        if token in ds.NEGATIONS:
            pending.append({"type": "negation", "token": token})
            negated = True
        elif token in ds.AMPLIFIERS:
            mult = ds.AMPLIFIERS[token]
            amplifier = min(amplifier * mult, 3.0)
            pending.append({"type": "amplifier", "token": token, "mult": mult})
        elif token in ds.EMOJI_SCORES:
            delta = float(ds.EMOJI_SCORES[token])
            score += delta
            pending.append({"type": "emoji", "token": token, "delta": delta})
            steps.append({"tokens": pending[:], "delta": delta, "score": int(score)})
            pending, amplifier, negated = [], 1.0, False
        elif token in ds.POSITIVE_WORDS:
            sig = WordSignal(ds.POSITIVE_WORDS[token]).name
            w = ds.SIGNAL_WEIGHTS[sig]
            delta = -(w * amplifier) if negated else (w * amplifier)
            score += delta
            pending.append({"type": "positive", "token": token, "sig": sig, "weight": w, "amp": amplifier, "negated": negated, "delta": delta})
            steps.append({"tokens": pending[:], "delta": delta, "score": int(score)})
            pending, amplifier, negated = [], 1.0, False
        elif token in ds.NEGATIVE_WORDS:
            sig = WordSignal(ds.NEGATIVE_WORDS[token]).name
            w = ds.SIGNAL_WEIGHTS[sig]
            delta = (w * amplifier) if negated else -(w * amplifier)
            score += delta
            pending.append({"type": "negative", "token": token, "sig": sig, "weight": w, "amp": amplifier, "negated": negated, "delta": delta})
            steps.append({"tokens": pending[:], "delta": delta, "score": int(score)})
            pending, amplifier, negated = [], 1.0, False
        else:
            pending.append({"type": "skip", "token": token})

    if pending:
        steps.append({"tokens": pending, "delta": 0, "score": int(score)})

    return steps, int(score)


_CHIP_STYLES = {
    "skip":      "background:#e0e0e0;color:#555",
    "negation":  "background:#ef5350;color:#fff",
    "amplifier": "background:#ff9800;color:#fff",
    "positive":  "background:#4caf50;color:#fff",
    "negative":  "background:#f44336;color:#fff",
    "emoji":     "background:#2196f3;color:#fff",
}
_CHIP_LABELS = {
    "skip": "", "negation": "NEGATE", "amplifier": "AMP",
    "positive": "POS", "negative": "NEG", "emoji": "EMOJI",
}

def render_walkthrough(sentence: str):
    steps, final_score = build_walkthrough(sentence)
    if not steps:
        st.info("No tokens found.")
        return

    html = '<div style="font-family:monospace;font-size:0.9em;line-height:2">'
    html += '<div style="margin-bottom:8px"><b>Start score: 50</b></div>'

    for step in steps:
        html += '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-bottom:2px">'
        for t in step["tokens"]:
            style = _CHIP_STYLES[t["type"]]
            chip = f'<span style="padding:2px 8px;border-radius:12px;{style}">{t["token"]}</span>'

            if t["type"] == "amplifier":
                chip += f'<span style="color:#ff9800;font-size:0.8em">×{t["mult"]}</span>'
            elif t["type"] in ("positive", "negative"):
                neg_str = " NEGATED" if t["negated"] else ""
                amp_str = f" ×{t['amp']:.1f}" if t["amp"] != 1.0 else ""
                chip += f'<span style="color:#888;font-size:0.8em"> {t["sig"]}({t["weight"]}){amp_str}{neg_str}</span>'
            elif t["type"] == "emoji":
                chip += f'<span style="color:#888;font-size:0.8em"> {t["delta"]:+.0f}</span>'

            html += chip + " "

        # flush line
        is_flush = step["tokens"] and step["tokens"][-1]["type"] in ("positive", "negative", "emoji")
        if is_flush:
            delta = step["delta"]
            color = "#4caf50" if delta > 0 else ("#f44336" if delta < 0 else "#888")
            html += (
                f'<span style="color:{color};font-weight:bold">'
                f'{"+" if delta >= 0 else ""}{delta:.1f}'
                f'</span>'
                f' → <b>score: {step["score"]}</b>'
                f' <span style="color:#bbb;font-size:0.8em">↺ reset</span>'
            )

        html += "</div>"

    # final verdict
    thresholds = ds.LABEL_THRESHOLDS
    if final_score >= thresholds["positive_above"]:
        verdict, vcolor = "positive", "#4caf50"
    elif final_score <= thresholds["negative_below"]:
        verdict, vcolor = "negative", "#f44336"
    else:
        verdict, vcolor = "mixed", "#ff9800"

    html += (
        f'<div style="margin-top:12px;font-size:1.1em">'
        f'<b>Final score: {final_score}</b> → '
        f'<span style="color:{vcolor};font-weight:bold">{verdict}</span>'
        f'</div>'
    )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Training Data
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Training Data")

    ITEMS_PER_PAGE = 10
    total = len(ds.SAMPLE_POSTS)
    total_pages = max(1, (total + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)

    # clamp page in case training set shrank
    st.session_state.training_page = min(st.session_state.training_page, total_pages)

    page = st.session_state.training_page
    start = (page - 1) * ITEMS_PER_PAGE
    end = min(start + ITEMS_PER_PAGE, total)

    df = pd.DataFrame({
        "#": range(start + 1, end + 1),
        "Post": ds.SAMPLE_POSTS[start:end],
        "Label": ds.TRUE_LABELS[start:end],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    # pagination controls
    pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
    if pcol1.button("← Prev", disabled=(page <= 1), use_container_width=True):
        st.session_state.training_page -= 1
        st.rerun()
    pcol2.markdown(
        f'<p style="text-align:center;margin:6px 0">Page {page} of {total_pages} &nbsp;·&nbsp; {total} examples</p>',
        unsafe_allow_html=True,
    )
    if pcol3.button("Next →", disabled=(page >= total_pages), use_container_width=True):
        st.session_state.training_page += 1
        st.rerun()

    st.divider()

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Add Training Example")
        with st.form("add_post_form"):
            new_post = st.text_input("Post / sentence")
            new_label = st.selectbox("Label", ["positive", "negative", "neutral", "mixed"])
            if st.form_submit_button("Add Example"):
                if new_post.strip():
                    add_sample_post(new_post.strip(), new_label)
                    st.success(f'Added "{new_post}" → {new_label}')
                    st.rerun()
                else:
                    st.warning("Post cannot be empty.")

    with right:
        st.subheader("Train ML Model")
        st.write(f"Current training set size: **{len(ds.SAMPLE_POSTS)}**")
        if st.button("Train / Retrain ML Model", type="primary", use_container_width=True):
            if len(ds.SAMPLE_POSTS) < 2:
                st.error("Need at least 2 training examples.")
            else:
                try:
                    vec, model = train_ml_model(ds.SAMPLE_POSTS, ds.TRUE_LABELS)
                    st.session_state.ml_vectorizer = vec
                    st.session_state.ml_model = model
                    st.session_state.ml_model_trained_at = datetime.now()
                    st.success(f"Trained on {len(ds.SAMPLE_POSTS)} examples.")
                except Exception as e:
                    st.error(str(e))

        if st.session_state.ml_model is not None:
            st.info("ML model is ready.")
        else:
            st.warning("ML model not trained yet.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Rule-Based Designer
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Rule-Based System Designer")

    # ── Section 1: Classification Rules ────────────────────────────────────
    st.subheader("Classification Rules")
    st.caption(
        "The analyzer scores every sentence starting at 50 (neutral). "
        "Words push the score up or down. You decide what score counts as positive, negative, or mixed."
    )

    thresh_col1, thresh_col2, thresh_col3 = st.columns(3)
    pos_thresh = thresh_col1.number_input(
        "Positive when score is above",
        min_value=51, max_value=100,
        value=ds.LABEL_THRESHOLDS["positive_above"],
        step=1,
    )
    neg_thresh = thresh_col2.number_input(
        "Negative when score is below",
        min_value=0, max_value=49,
        value=ds.LABEL_THRESHOLDS["negative_below"],
        step=1,
    )
    thresh_col3.markdown("<br>", unsafe_allow_html=True)
    thresh_col3.info(f"Mixed: scores between **{neg_thresh}** and **{pos_thresh}**")

    if st.button("Apply Classification Rules"):
        try:
            set_label_thresholds(int(pos_thresh), int(neg_thresh))
            st.session_state.rule_updated_at = datetime.now()
            st.success("Classification rules updated.")
        except ValueError as e:
            st.error(str(e))

    st.divider()

    # ── Section 2: Signal Strength Designer ────────────────────────────────
    st.subheader("Signal Strength")
    st.caption(
        "Each word in your lists is tagged WEAK, MEDIUM, or STRONG. "
        "Set how many points each level adds or subtracts from the score."
    )

    sig_col1, sig_col2, sig_col3 = st.columns(3)
    weak_val = sig_col1.slider(
        "WEAK  ±", min_value=1, max_value=30,
        value=ds.SIGNAL_WEIGHTS["WEAK"],
        help="e.g. 'okay', 'fine', 'tired'",
    )
    med_val = sig_col2.slider(
        "MEDIUM  ±", min_value=1, max_value=30,
        value=ds.SIGNAL_WEIGHTS["MEDIUM"],
        help="e.g. 'happy', 'sad', 'excited'",
    )
    strong_val = sig_col3.slider(
        "STRONG  ±", min_value=1, max_value=30,
        value=ds.SIGNAL_WEIGHTS["STRONG"],
        help="e.g. 'love', 'hate', 'amazing'",
    )

    if st.button("Apply Signal Weights"):
        set_signal_weights(weak_val, med_val, strong_val)
        st.session_state.rule_updated_at = datetime.now()
        st.success(f"Weights set — WEAK ±{weak_val}, MEDIUM ±{med_val}, STRONG ±{strong_val}")

    st.divider()

    # ── Section 3: Score Walkthrough ────────────────────────────────────────
    st.subheader("Score Walkthrough")
    st.caption(
        "Type any sentence to see how the rule-based scorer processes it token by token. "
        "When a sentiment word is hit the modifiers are consumed and the score updates — then the stack resets."
    )
    demo_sentence = st.text_input("Sentence to trace", value="I am not very happy today", key="walkthrough_input")
    if demo_sentence.strip():
        render_walkthrough(demo_sentence.strip())

    st.divider()

    # ── Section 4: Word Lists (editable tables) ─────────────────────────────
    st.subheader("Word Lists")
    st.caption("Click any cell to edit. Use the last empty row to add a new word. Press Save to apply.")

    word_col1, word_col2 = st.columns(2)

    with word_col1:
        st.markdown("**Positive Words**")
        pos_df = pd.DataFrame(
            [{"Word": w, "Signal": WordSignal(v).name} for w, v in ds.POSITIVE_WORDS.items()]
        )
        edited_pos = st.data_editor(
            pos_df,
            column_config={
                "Signal": st.column_config.SelectboxColumn(
                    options=["WEAK", "MEDIUM", "STRONG"], required=True
                )
            },
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="pos_editor",
        )
        if st.button("Save Positive Words"):
            ds.POSITIVE_WORDS.clear()
            for _, row in edited_pos.dropna(subset=["Word"]).iterrows():
                word = str(row["Word"]).strip().lower()
                if word:
                    ds.POSITIVE_WORDS[word] = WordSignal[row["Signal"]]
            st.session_state.rule_updated_at = datetime.now()
            st.success("Positive words saved.")
            st.rerun()

    with word_col2:
        st.markdown("**Negative Words**")
        neg_df = pd.DataFrame(
            [{"Word": w, "Signal": WordSignal(v).name} for w, v in ds.NEGATIVE_WORDS.items()]
        )
        edited_neg = st.data_editor(
            neg_df,
            column_config={
                "Signal": st.column_config.SelectboxColumn(
                    options=["WEAK", "MEDIUM", "STRONG"], required=True
                )
            },
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="neg_editor",
        )
        if st.button("Save Negative Words"):
            ds.NEGATIVE_WORDS.clear()
            for _, row in edited_neg.dropna(subset=["Word"]).iterrows():
                word = str(row["Word"]).strip().lower()
                if word:
                    ds.NEGATIVE_WORDS[word] = WordSignal[row["Signal"]]
            st.session_state.rule_updated_at = datetime.now()
            st.success("Negative words saved.")
            st.rerun()

    st.divider()

    # ── Section 4: Modifiers ────────────────────────────────────────────────
    st.subheader("Modifiers")
    mod_col1, mod_col2, mod_col3 = st.columns(3)

    with mod_col1:
        st.markdown("**Amplifiers** — multiply word impact")
        amp_df = pd.DataFrame([{"Word": w, "Multiplier": v} for w, v in ds.AMPLIFIERS.items()])
        edited_amp = st.data_editor(
            amp_df,
            column_config={"Multiplier": st.column_config.NumberColumn(min_value=1.0, max_value=5.0, step=0.1)},
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="amp_editor",
        )
        if st.button("Save Amplifiers"):
            ds.AMPLIFIERS.clear()
            for _, row in edited_amp.dropna(subset=["Word"]).iterrows():
                word = str(row["Word"]).strip().lower()
                if word:
                    ds.AMPLIFIERS[word] = float(row["Multiplier"])
            st.session_state.rule_updated_at = datetime.now()
            st.success("Amplifiers saved.")
            st.rerun()

    with mod_col2:
        st.markdown("**Emojis** — direct score offset")
        emoji_df = pd.DataFrame([{"Emoji": e, "Score": s} for e, s in ds.EMOJI_SCORES.items()])
        edited_emoji = st.data_editor(
            emoji_df,
            column_config={"Score": st.column_config.NumberColumn(min_value=-15, max_value=15, step=1)},
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="emoji_editor",
        )
        if st.button("Save Emojis"):
            ds.EMOJI_SCORES.clear()
            for _, row in edited_emoji.dropna(subset=["Emoji"]).iterrows():
                ds.EMOJI_SCORES[str(row["Emoji"])] = int(row["Score"])
            st.session_state.rule_updated_at = datetime.now()
            st.success("Emojis saved.")
            st.rerun()

    with mod_col3:
        st.markdown("**Negations** — flip word polarity")
        neg_set_df = pd.DataFrame([{"Word": w} for w in sorted(ds.NEGATIONS)])
        edited_neg_set = st.data_editor(
            neg_set_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="neg_set_editor",
        )
        if st.button("Save Negations"):
            ds.NEGATIONS.clear()
            for _, row in edited_neg_set.dropna(subset=["Word"]).iterrows():
                word = str(row["Word"]).strip().lower()
                if word:
                    ds.NEGATIONS.add(word)
            st.session_state.rule_updated_at = datetime.now()
            st.success("Negations saved.")
            st.rerun()


# ── Default test batch — 3 examples of each label ────────────────────────────
DEFAULT_TEST_BATCH = [
    ("no cap that genuinely hit different 😍", "positive"),
    ("Aced my exam, feeling fantastic 😄", "positive"),
    ("I never feel this relaxed", "positive"),
    ("I absolutely love waiting in line for an hour 🙃", "negative"),
    ("so frustrated I could scream 😤", "negative"),
    ("I'm not happy about this at all", "negative"),
    ("today was okay I guess", "neutral"),
    ("It is what it is honestly", "neutral"),
    ("Not much happened today, pretty quiet", "neutral"),
    ("lowkey stressed but also kinda thriving?", "mixed"),
    ("I'm excited but also really nervous", "mixed"),
    ("kinda vibing kinda not tbh 😅", "mixed"),
]

# ── Auto-load default test set on first load ────────────────────────────────
if not st.session_state.test_posts:
    st.session_state.test_posts = [e[0] for e in DEFAULT_TEST_BATCH]
    st.session_state.test_labels = [e[1] for e in DEFAULT_TEST_BATCH]


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Testing & Evaluation
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Testing Set & Evaluation")

    # ── Quick Inference ──────────────────────────────────────────────────
    st.subheader("Try a Sentence")
    st.caption("Type any sentence below to instantly see what both models predict — no label needed.")
    _qi_col1, _qi_col2 = st.columns([4, 1])
    with _qi_col1:
        _qi_sentence = st.text_input(
            "Your sentence",
            placeholder='e.g. "I\'m so happy but also kinda nervous about this"',
            label_visibility="collapsed",
            key="qi_sentence_input",
        )
    with _qi_col2:
        _qi_predict = st.button("Predict →", type="primary", use_container_width=True, key="qi_predict_btn")

    if _qi_sentence.strip() and _qi_predict:
        _qi_rule_pred = MoodAnalyzer().predict_label(_qi_sentence.strip())
        _qi_ml_pred = None
        if st.session_state.ml_model is not None:
            _qi_ml_pred = predict_single_text(
                _qi_sentence.strip(), st.session_state.ml_vectorizer, st.session_state.ml_model
            )
        _qc1, _qc2 = st.columns(2)
        _qc1.metric("Rule-Based Prediction", _qi_rule_pred)
        if _qi_ml_pred is not None:
            _qc2.metric("ML Model Prediction", _qi_ml_pred)
        else:
            _qc2.info("Train the ML model first (Training Data tab).")
        with st.expander("See rule-based reasoning"):
            render_walkthrough(_qi_sentence.strip())

    st.divider()

    left, right = st.columns([3, 1])

    with right:
        st.subheader("Add Test Example")
        with st.form("add_test"):
            test_post = st.text_input("Sentence")
            test_label = st.selectbox("True Label", ["positive", "negative", "neutral", "mixed"])
            if st.form_submit_button("Add to Test Set", use_container_width=True):
                if test_post.strip():
                    st.session_state.test_posts.append(test_post.strip())
                    st.session_state.test_labels.append(test_label)
                    st.rerun()
                else:
                    st.warning("Sentence cannot be empty.")

        st.subheader("Default Batches")
        st.caption("Loads 3 examples of each label (12 total).")
        if st.button("Load Sample Batch", use_container_width=True):
            st.session_state.test_posts = [e[0] for e in DEFAULT_TEST_BATCH]
            st.session_state.test_labels = [e[1] for e in DEFAULT_TEST_BATCH]
            st.session_state.eval_results = None
            st.rerun()
        if st.button("Clear Test Set", use_container_width=True):
            st.session_state.test_posts = []
            st.session_state.test_labels = []
            st.session_state.eval_results = None
            st.rerun()

    with left:
        st.subheader(f"Test Set ({len(st.session_state.test_posts)} examples)")
        if not st.session_state.test_posts:
            st.info("No test examples yet. Add some on the right.")
        else:
            for i, (post, label) in enumerate(
                zip(st.session_state.test_posts, st.session_state.test_labels)
            ):
                c1, c2, c3 = st.columns([6, 1, 1])
                c1.write(post)
                c2.markdown(f"`{label}`")
                if c3.button("Delete", key=f"del_{i}"):
                    st.session_state.test_posts.pop(i)
                    st.session_state.test_labels.pop(i)
                    st.rerun()

    st.divider()

    if st.button("Run Batch Evaluation", type="primary", use_container_width=False):
        if not st.session_state.test_posts:
            st.warning("Add test examples first.")
        else:
            rule_analyzer = MoodAnalyzer()
            rule_preds = [rule_analyzer.predict_label(p) for p in st.session_state.test_posts]
            rule_acc = accuracy_score(st.session_state.test_labels, rule_preds)

            ml_preds = None
            ml_acc = None
            if st.session_state.ml_model is not None:
                ml_preds = [
                    predict_single_text(p, st.session_state.ml_vectorizer, st.session_state.ml_model)
                    for p in st.session_state.test_posts
                ]
                ml_acc = accuracy_score(st.session_state.test_labels, ml_preds)

            st.session_state.eval_results = {
                "posts": list(st.session_state.test_posts),
                "true_labels": list(st.session_state.test_labels),
                "rule_preds": rule_preds,
                "rule_acc": rule_acc,
                "ml_preds": ml_preds,
                "ml_acc": ml_acc,
            }

    r = st.session_state.eval_results
    if r:
        # Results table — always show both columns side by side
        st.subheader("Predictions")
        ml_col_values = r["ml_preds"] if r["ml_preds"] is not None else ["(not trained)"] * len(r["posts"])
        results_df = pd.DataFrame({
            "Sentence": r["posts"],
            "True Label": r["true_labels"],
            "Rule-Based": r["rule_preds"],
            "ML Model": ml_col_values,
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Per-row walkthroughs
        st.subheader("Rule-Based Walkthrough")
        for i, (post, true_label, rule_pred) in enumerate(
            zip(r["posts"], r["true_labels"], r["rule_preds"])
        ):
            correct = rule_pred == true_label
            icon = "✓" if correct else "✗"
            label_color = "green" if correct else "red"
            header = f'{icon} "{post}"  —  predicted: **{rule_pred}**  |  true: **{true_label}**'
            with st.expander(header, expanded=not correct):
                render_walkthrough(post)

        # ML per-row results
        if r["ml_preds"] is not None:
            st.subheader("ML Model Results")
            for post, true_label, ml_pred in zip(r["posts"], r["true_labels"], r["ml_preds"]):
                correct = ml_pred == true_label
                icon = "✓" if correct else "✗"
                st.markdown(
                    f'{icon} "{post}"  —  predicted: **{ml_pred}**  |  true: **{true_label}**'
                )

        # Accuracy metrics
        st.subheader("Accuracy")
        m1, m2 = st.columns(2)
        m1.metric("Rule-Based", f"{r['rule_acc']:.0%}")
        if r["ml_acc"] is not None:
            m2.metric("ML Model", f"{r['ml_acc']:.0%}")
        else:
            m2.info("Train the ML model first (Training Data tab).")

        ts_col1, ts_col2 = st.columns(2)
        if st.session_state.rule_updated_at is not None:
            ts_col1.caption(f"Rule last updated: {st.session_state.rule_updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if st.session_state.ml_model_trained_at is not None:
            ts_col2.caption(f"ML model last trained: {st.session_state.ml_model_trained_at.strftime('%Y-%m-%d %H:%M:%S')}")

        # Learning curve
        if st.session_state.ml_model is not None and len(ds.SAMPLE_POSTS) >= 3:
            st.subheader("Learning Curve")
            train_sizes = list(range(2, len(ds.SAMPLE_POSTS) + 1))
            lc_accs = []
            for size in train_sizes:
                try:
                    v, m = train_ml_model(ds.SAMPLE_POSTS[:size], ds.TRUE_LABELS[:size])
                    preds = [predict_single_text(p, v, m) for p in r["posts"]]
                    lc_accs.append(accuracy_score(r["true_labels"], preds))
                except Exception:
                    lc_accs.append(None)

            valid = [(s, a) for s, a in zip(train_sizes, lc_accs) if a is not None]
            if valid:
                xs, ys = zip(*valid)
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.plot(xs, ys, marker="o", label="ML Model", color="#1f77b4")
                ax.axhline(
                    r["rule_acc"],
                    linestyle="--",
                    color="orange",
                    label=f"Rule-Based ({r['rule_acc']:.0%})",
                )
                ax.set_xlabel("Training Set Size")
                ax.set_ylabel("Test Accuracy")
                ax.set_title("ML Model Learning Curve vs Rule-Based Baseline")
                ax.set_ylim(0, 1.05)
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Active Learning
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Active Learning")
    st.caption(
        "Cluster a batch of unlabeled posts, label a few examples from each cluster, "
        "then retrain the ML model with an 80/20 train/test split."
    )

    # ── Progress indicator ────────────────────────────────────────────────
    # Determine current mode label for display
    # Cold start = round 1 and no user-labeled data added beyond the seed baseline
    _is_cold_start = (
        st.session_state.al_round == 1
        and len(ds.SAMPLE_POSTS) <= st.session_state.al_original_count
    )

    if _is_cold_start:
        st.caption(
            "**Round 0 — Cold Start (Diversity Sampling):** No training data yet. "
            "Cluster the full unlabeled dataset to get a diverse sample, label those examples, "
            "and train your first baseline model."
        )
        _step_labels = ["1 · Load", "2 · Cluster", "3 · Label", "4 · Build Baseline"]
    else:
        _round_num = st.session_state.al_round
        st.caption(
            f"**Round {_round_num} — Active Learning (Uncertainty Sampling):** "
            "The model selects the examples it is most confused about. "
            "Label those to improve it as fast as possible."
        )
        _step_labels = ["1 · Load", "2 · Select Examples", "3 · Label", "4 · Retrain"]

    _step_cols = st.columns(4)
    for _i, (_col, _lbl) in enumerate(zip(_step_cols, _step_labels)):
        _active = st.session_state.al_step == _i
        _done = st.session_state.al_step > _i
        _color = "#1f77b4" if _active else ("#4caf50" if _done else "#bbb")
        _weight = "bold" if _active else "normal"
        _col.markdown(
            f'<div style="text-align:center;color:{_color};font-weight:{_weight}'
            f';border-bottom:3px solid {_color};padding-bottom:4px">{_lbl}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Helper: reset all active learning state ───────────────────────────
    def _al_reset():
        st.session_state.al_step = 0
        st.session_state.al_raw_texts = []
        st.session_state.al_X = None
        st.session_state.al_cluster_labels = None
        st.session_state.al_elbow_data = None
        st.session_state.al_sample_indices = None
        st.session_state.al_pending_labels = {}
        st.session_state.al_committed_indices = set()
        st.session_state.al_retrain_results = None
        st.session_state.al_round = 1
        st.session_state.al_accuracy_before = None
        st.session_state.al_mode = "diversity"
        st.session_state.al_uncertainty_indices = None
        st.session_state.al_baseline_rule_acc = None
        st.session_state.al_baseline_ml_acc = None

    # ════════════════════════════════════════════════════════════════════════
    # STEP 1 — Load unlabeled posts
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("Step 1 — Load Posts")

    _al1_col1, _al1_col2 = st.columns(2)

    with _al1_col1:
        st.markdown("**Built-in Examples**")
        st.caption("100 diverse sentences covering a range of moods and contexts.")
        if st.button("Load 100 Diverse Examples", type="primary"):
            st.session_state.al_raw_texts = list(UNLABELED_EXAMPLES)
            st.session_state.al_step = 1
            st.session_state.al_X = None
            st.session_state.al_cluster_labels = None
            st.session_state.al_elbow_data = None
            st.session_state.al_sample_indices = None
            st.session_state.al_pending_labels = {}
            st.session_state.al_committed_indices = set()
            st.session_state.al_retrain_results = None
            st.session_state.al_round = 1
            st.session_state.al_accuracy_before = None
            st.rerun()

        # Auto-load on cold start: if no training data and nothing loaded yet, pre-load automatically
        if _is_cold_start and st.session_state.al_step == 0 and not st.session_state.al_raw_texts:
            st.session_state.al_raw_texts = list(UNLABELED_EXAMPLES)
            st.session_state.al_step = 1
            st.session_state.al_mode = "diversity"
            st.rerun()

    with _al1_col2:
        st.markdown("**Upload Your Own CSV**")
        st.caption(
            "CSV must have a **sentence** column (or use the first column). "
            "Add a **label** column (`positive`, `negative`, `neutral`, `mixed`) "
            "to add labeled rows directly to training data; unlabeled rows enter the clustering workflow."
        )
        _al_csv = st.file_uploader("Choose CSV file", type=["csv"], key="al_csv_uploader")
        if _al_csv is not None:
            try:
                _al_csv_df = pd.read_csv(_al_csv)
                # Detect sentence column
                _sentence_col = None
                for _c in _al_csv_df.columns:
                    if _c.strip().lower() in ("sentence", "text", "post", "sentences", "texts"):
                        _sentence_col = _c
                        break
                if _sentence_col is None:
                    _sentence_col = _al_csv_df.columns[0]
                # Detect label column
                _label_col = None
                for _c in _al_csv_df.columns:
                    if _c.strip().lower() in ("label", "labels", "sentiment", "mood"):
                        _label_col = _c
                        break
                _al_csv_df = _al_csv_df.dropna(subset=[_sentence_col])
                _all_sentences = _al_csv_df[_sentence_col].astype(str).str.strip().tolist()

                if _label_col is not None:
                    _valid_labels = {"positive", "negative", "neutral", "mixed"}
                    _labeled_mask = _al_csv_df[_label_col].astype(str).str.strip().str.lower().isin(_valid_labels)
                    _labeled_df = _al_csv_df[_labeled_mask].copy()
                    _unlabeled_sentences = _al_csv_df[~_labeled_mask][_sentence_col].astype(str).str.strip().tolist()
                    st.markdown(
                        f"Found **{len(_labeled_df)}** labeled rows and **{len(_unlabeled_sentences)}** unlabeled rows."
                    )
                    if len(_labeled_df) > 0:
                        if st.button(f"Add {len(_labeled_df)} Labeled Rows to Training Data", key="al_csv_add_training"):
                            _added = 0
                            for _, _row in _labeled_df.iterrows():
                                _s = str(_row[_sentence_col]).strip()
                                _l = str(_row[_label_col]).strip().lower()
                                if _s:
                                    add_sample_post(_s, _l)
                                    _added += 1
                            st.success(f"Added {_added} examples to training data. Go to Training Data tab to retrain.")
                    if len(_unlabeled_sentences) > 0:
                        if st.button(f"Load {len(_unlabeled_sentences)} Unlabeled Rows for Active Learning", key="al_csv_load_unlabeled"):
                            st.session_state.al_raw_texts = _unlabeled_sentences
                            st.session_state.al_step = 1
                            st.session_state.al_X = None
                            st.session_state.al_cluster_labels = None
                            st.session_state.al_elbow_data = None
                            st.session_state.al_sample_indices = None
                            st.session_state.al_pending_labels = {}
                            st.session_state.al_committed_indices = set()
                            st.session_state.al_retrain_results = None
                            st.session_state.al_round = 1
                            st.session_state.al_accuracy_before = None
                            st.rerun()
                else:
                    st.markdown(f"Found **{len(_all_sentences)}** sentences (no label column detected).")
                    if st.button(f"Load {len(_all_sentences)} Sentences for Active Learning", key="al_csv_load_all"):
                        st.session_state.al_raw_texts = _all_sentences
                        st.session_state.al_step = 1
                        st.session_state.al_X = None
                        st.session_state.al_cluster_labels = None
                        st.session_state.al_elbow_data = None
                        st.session_state.al_sample_indices = None
                        st.session_state.al_pending_labels = {}
                        st.session_state.al_committed_indices = set()
                        st.session_state.al_retrain_results = None
                        st.session_state.al_round = 1
                        st.session_state.al_accuracy_before = None
                        st.rerun()
            except Exception as _csv_err:
                st.error(f"Could not read CSV: {_csv_err}")

    if st.session_state.al_step >= 1:
        st.success(f"{len(st.session_state.al_raw_texts)} posts loaded. Proceed to Step 2 to cluster them.")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2 — Cluster (cold start) or Uncertainty Sampling (rounds 1+)
    # ════════════════════════════════════════════════════════════════════════
    if st.session_state.al_step >= 1:
        st.divider()

        _remaining_indices = [
            i for i in range(len(st.session_state.al_raw_texts))
            if i not in st.session_state.al_committed_indices
        ]
        _remaining_texts = [st.session_state.al_raw_texts[i] for i in _remaining_indices]
        _n_texts = len(_remaining_texts)

        # ── COLD START: diversity sampling via clustering ──────────────────
        if _is_cold_start:
            st.subheader("Step 2 — Cluster for Diversity (Cold Start)")
            st.caption(
                "We cluster all unlabeled examples so you label a few from each group. "
                "This gives your first model a diverse foundation to learn from."
            )

            _k_max = min(10, _n_texts // 2)
            st.markdown("**1. Compute the elbow curve to choose K**")
            st.caption("The elbow is where inertia stops dropping sharply — that is your best K.")

            if st.button("Compute Elbow Curve", key="al_elbow_btn"):
                with st.spinner("Computing inertia for k = 2…10…"):
                    st.session_state.al_elbow_data = _cached_elbow(
                        tuple(_remaining_texts), min(10, _n_texts // 2)
                    )
                st.rerun()

            if st.session_state.al_elbow_data:
                _ks, _inertias = zip(*st.session_state.al_elbow_data)
                _fig_elbow, _ax_elbow = plt.subplots(figsize=(8, 3))
                _ax_elbow.plot(_ks, _inertias, marker="o", color="#1f77b4")
                _ax_elbow.set_xlabel("K (number of clusters)")
                _ax_elbow.set_ylabel("Inertia")
                _ax_elbow.set_title("Elbow Curve — pick K at the bend")
                _ax_elbow.grid(True, alpha=0.3)
                st.pyplot(_fig_elbow)
                _k = st.slider("Number of clusters (K)", 2, max(2, _k_max), min(4, _k_max), key="al_k")
            else:
                st.info("Compute the elbow curve first to choose K.")
                _k = min(4, _k_max)

            _n_per_cluster = st.slider("Examples to label per cluster", 1, 5, 2, key="al_npc")

            if st.button("Run Clustering", type="primary", disabled=st.session_state.al_elbow_data is None):
                with st.spinner("Vectorizing and clustering…"):
                    _, _X = _cached_vectorize(tuple(_remaining_texts))
                    _labels = cluster_texts(_X, "kmeans", k=_k)
                    st.session_state.al_cluster_labels = _labels
                    _raw_samples = sample_from_clusters(_remaining_texts, _labels, _X, "kmeans", _n_per_cluster)
                    # Remap cluster sample indices back to original al_raw_texts indices
                    _mapped = {
                        cid: [_remaining_indices[local_idx] for local_idx in local_idxs]
                        for cid, local_idxs in _raw_samples.items()
                    }
                    st.session_state.al_sample_indices = _mapped
                    st.session_state.al_step = 2
                    st.session_state.al_mode = "diversity"
                st.rerun()

            if st.session_state.al_step >= 2 and st.session_state.al_cluster_labels is not None:
                _lbl_arr = st.session_state.al_cluster_labels
                _real_clusters = [c for c in set(_lbl_arr.tolist()) if c != -1]
                _n_noise = int((_lbl_arr == -1).sum())
                st.success(
                    f"Found **{len(_real_clusters)}** cluster(s)"
                    + (f" · {_n_noise} noise post(s)" if _n_noise else "")
                )
                import collections as _coll
                _counts = _coll.Counter(_lbl_arr.tolist())
                _bar_df = pd.DataFrame(
                    [
                        {"Cluster": ("Noise" if k == -1 else f"Cluster {k}"), "Posts": v}
                        for k, v in sorted(_counts.items())
                    ]
                )
                st.bar_chart(_bar_df.set_index("Cluster"))

        # ── ACTIVE LEARNING ROUNDS: uncertainty sampling ───────────────────
        else:
            st.subheader(f"Step 2 — Select Uncertain Examples (Round {st.session_state.al_round})")
            st.caption(
                "The model picks the examples it is least confident about. "
                "Labeling these teaches the model more per label than random selection."
            )

            if st.session_state.al_round > 1:
                st.info(
                    f"**{_n_texts}** unlabeled examples remain "
                    f"({len(st.session_state.al_committed_indices)} labeled so far)."
                )

            _n_uncertain = st.slider(
                "Examples to label this round",
                min_value=3,
                max_value=min(20, _n_texts),
                value=min(8, _n_texts),
                key="al_n_uncertain",
            )

            if st.button("Find Most Uncertain Examples", type="primary"):
                with st.spinner("Running model on unlabeled pool…"):
                    _uncertain_local = uncertainty_sampling(
                        ds.SAMPLE_POSTS,
                        ds.TRUE_LABELS,
                        _remaining_texts,
                        _n_uncertain,
                    )
                    # Map local indices back to al_raw_texts indices
                    st.session_state.al_uncertainty_indices = [
                        _remaining_indices[i] for i in _uncertain_local
                    ]
                    # Store as al_sample_indices in the same format as clustering uses
                    # so the labeling step (Step 3) can remain unchanged
                    st.session_state.al_sample_indices = {
                        0: st.session_state.al_uncertainty_indices
                    }
                    st.session_state.al_step = 2
                    st.session_state.al_mode = "uncertainty"
                st.rerun()

            # Show a confidence chart if model exists
            if st.session_state.ml_model is not None and _n_texts > 0:
                with st.expander("Preview model confidence on remaining unlabeled examples"):
                    import numpy as np
                    _vec_tmp = st.session_state.ml_vectorizer
                    _mdl_tmp = st.session_state.ml_model
                    _X_preview = _vec_tmp.transform(_remaining_texts)
                    _proba_preview = _mdl_tmp.predict_proba(_X_preview)
                    _confidence = _proba_preview.max(axis=1)
                    _sorted_conf = sorted(enumerate(_confidence), key=lambda x: x[1])
                    _conf_data = pd.DataFrame({
                        "Text": [_remaining_texts[i][:60] + "…" if len(_remaining_texts[i]) > 60 else _remaining_texts[i] for i, _ in _sorted_conf[:15]],
                        "Max Confidence": [f"{c:.0%}" for _, c in _sorted_conf[:15]],
                    })
                    st.caption("Top 15 most uncertain (lowest confidence) — these are what the model will select.")
                    st.dataframe(_conf_data, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 3 — Label
    # ════════════════════════════════════════════════════════════════════════
    if st.session_state.al_step >= 2 and st.session_state.al_sample_indices is not None:
        st.divider()
        if st.session_state.al_mode == "diversity":
            st.subheader("Step 3 — Label Cluster Representatives")
            st.caption(
                "Each group below contains the most representative examples from one cluster. "
                "Label at least one per group, then commit."
            )
        else:
            st.subheader("Step 3 — Label Uncertain Examples")
            st.caption(
                "These are the examples the model is least confident about. "
                "Your labels here will improve it the most."
            )

        _analyzer = MoodAnalyzer()
        _pending = st.session_state.al_pending_labels
        _raw = st.session_state.al_raw_texts

        _label_options = ["(skip)", "positive", "negative", "neutral", "mixed"]
        _label_colors = {
            "positive": "#4caf50",
            "negative": "#f44336",
            "neutral": "#2196f3",
            "mixed": "#ff9800",
        }

        _total_shown = sum(len(v) for v in st.session_state.al_sample_indices.values())
        _current_indices = {idx for idxs in st.session_state.al_sample_indices.values() for idx in idxs}
        _total_labeled = sum(1 for idx, lbl in _pending.items() if idx in _current_indices and lbl != "(skip)")
        st.progress(
            min(_total_labeled / _total_shown, 1.0) if _total_shown else 0,
            text=f"Labeled {_total_labeled} / {_total_shown}",
        )

        for _cid, _indices in st.session_state.al_sample_indices.items():
            if st.session_state.al_mode == "uncertainty":
                _expander_title = f"Uncertain Examples ({len(_indices)} examples)"
            else:
                _expander_title = "Noise / Outliers" if _cid == -1 else f"Cluster {_cid} — {len(_indices)} examples"
            with st.expander(_expander_title, expanded=True):
                for _idx in _indices:
                    _post = _raw[_idx]
                    _txt_col, _lbl_col, _preview_col = st.columns([5, 2, 3])
                    _txt_col.markdown(f"> {_post}")

                    _current = _pending.get(_idx, "(skip)")
                    _chosen = _lbl_col.selectbox(
                        "Label",
                        options=_label_options,
                        index=_label_options.index(_current),
                        key=f"al_label_{_idx}",
                        label_visibility="collapsed",
                    )
                    # Keep pending dict in sync (widget drives value)
                    st.session_state.al_pending_labels[_idx] = _chosen

                    # Live preview: rule-based and ML
                    if _chosen != "(skip)":
                        _rule_pred = _analyzer.predict_label(_post)
                        _rule_color = _label_colors.get(_rule_pred, "#888")
                        _preview_md = (
                            f'<span style="font-size:0.8em;color:#888">Rule: </span>'
                            f'<span style="color:{_rule_color};font-weight:bold;font-size:0.8em">{_rule_pred}</span>'
                        )
                        if st.session_state.ml_model is not None:
                            _ml_pred = predict_single_text(
                                _post,
                                st.session_state.ml_vectorizer,
                                st.session_state.ml_model,
                            )
                            _ml_color = _label_colors.get(_ml_pred, "#888")
                            _preview_md += (
                                f'<span style="font-size:0.8em;color:#888"> · ML: </span>'
                                f'<span style="color:{_ml_color};font-weight:bold;font-size:0.8em">{_ml_pred}</span>'
                            )
                        _preview_col.markdown(_preview_md, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Commit Labels to Training Set", type="primary"):
            _to_commit = {
                idx: lbl
                for idx, lbl in st.session_state.al_pending_labels.items()
                if lbl != "(skip)" and idx not in st.session_state.al_committed_indices
            }
            if len(_to_commit) < 2:
                st.warning("Label at least 2 examples before committing.")
            else:
                for _idx, _lbl in _to_commit.items():
                    ds.add_sample_post(st.session_state.al_raw_texts[_idx], _lbl)
                    st.session_state.al_committed_indices.add(_idx)
                st.session_state.al_step = 3
                st.rerun()

        # Show next-step options after a successful commit
        if st.session_state.al_step >= 3:
            _committed_so_far = len(st.session_state.al_committed_indices)
            _pool_total = len(st.session_state.al_raw_texts)
            _still_remaining = _pool_total - _committed_so_far

            st.success(
                f"Round {st.session_state.al_round} complete — "
                f"**{_committed_so_far}** labeled, **{_still_remaining}** still unlabeled."
            )

            _next_col1, _next_col2 = st.columns(2)

            if _still_remaining >= 5:
                if _next_col1.button(
                    f"Label Another Round ({_still_remaining} posts left)",
                    use_container_width=True,
                ):
                    st.session_state.al_round += 1
                    st.session_state.al_step = 1
                    st.session_state.al_X = None
                    st.session_state.al_cluster_labels = None
                    st.session_state.al_sample_indices = None
                    st.session_state.al_pending_labels = {}
                    st.session_state.al_mode = "uncertainty"
                    st.rerun()
            else:
                _next_col1.info("All posts labeled — ready to retrain.")

            if _next_col2.button("Proceed to Retrain →", type="primary", use_container_width=True):
                st.session_state.al_step = 4
                st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # STEP 4 — Retrain
    # ════════════════════════════════════════════════════════════════════════
    if st.session_state.al_step >= 4:
        st.divider()
        if st.session_state.al_mode == "diversity":
            st.subheader("Step 4 — Build Baseline")
            st.caption(
                "You have labeled your first batch using diversity sampling. "
                "Train both the rule-based and ML models and see where each one stands."
            )
        else:
            st.subheader(f"Step 4 — Retrain (Round {st.session_state.al_round})")
            st.caption("Retrain the ML model with your new labels and see how much it improved.")

        _orig = st.session_state.al_original_count
        _new = len(st.session_state.al_committed_indices)
        _total = len(ds.SAMPLE_POSTS)

        _mc1, _mc2, _mc3 = st.columns(3)
        _mc1.metric("Original Training Set", _orig)
        _mc2.metric("Newly Labeled", _new, delta=f"+{_new}")
        _mc3.metric("Total Training Set", _total)

        st.caption("Retraining replaces the ML model used across all tabs.")

        if not st.session_state.test_posts:
            st.warning(
                "No test set found. Add labeled examples in the **Testing & Evaluation** tab "
                "to see a before/after accuracy comparison."
            )

        if st.button("Retrain ML Model", type="primary"):
            # Capture accuracy on the existing test set BEFORE retraining
            if st.session_state.test_posts and st.session_state.ml_model is not None:
                _before_preds = [
                    predict_single_text(p, st.session_state.ml_vectorizer, st.session_state.ml_model)
                    for p in st.session_state.test_posts
                ]
                st.session_state.al_accuracy_before = accuracy_score(
                    st.session_state.test_labels, _before_preds
                )

            with st.spinner("Training…"):
                try:
                    _vec, _model = train_ml_model(ds.SAMPLE_POSTS, ds.TRUE_LABELS)
                    st.session_state.ml_vectorizer = _vec
                    st.session_state.ml_model = _model
                    st.session_state.ml_model_trained_at = datetime.now()
                    st.session_state.al_retrain_results = {"done": True}

                    # After retraining, if this was the cold start round, record baselines
                    if st.session_state.al_mode == "diversity" and st.session_state.al_baseline_ml_acc is None:
                        # Compute rule-based baseline on the test set if one exists
                        if st.session_state.test_posts:
                            _analyzer_base = MoodAnalyzer()
                            _rule_preds_base = [_analyzer_base.predict_label(p) for p in st.session_state.test_posts]
                            _rule_base_acc = accuracy_score(st.session_state.test_labels, _rule_preds_base)
                            st.session_state.al_baseline_rule_acc = _rule_base_acc

                        # Compute ML baseline
                        if st.session_state.test_posts and st.session_state.ml_model is not None:
                            _ml_preds_base = [
                                predict_single_text(p, st.session_state.ml_vectorizer, st.session_state.ml_model)
                                for p in st.session_state.test_posts
                            ]
                            _ml_base_acc = accuracy_score(st.session_state.test_labels, _ml_preds_base)
                            st.session_state.al_baseline_ml_acc = _ml_base_acc

                    st.rerun()
                except Exception as _e:
                    st.error(str(_e))

        if st.session_state.al_retrain_results and st.session_state.al_retrain_results.get("done"):
            # ── Before / After accuracy on test set ───────────────────────
            if st.session_state.test_posts:
                _eval_analyzer = MoodAnalyzer()
                _rule_preds = [_eval_analyzer.predict_label(p) for p in st.session_state.test_posts]
                _acc_rule = accuracy_score(st.session_state.test_labels, _rule_preds)

                _after_preds = [
                    predict_single_text(p, st.session_state.ml_vectorizer, st.session_state.ml_model)
                    for p in st.session_state.test_posts
                ]
                _acc_after = accuracy_score(st.session_state.test_labels, _after_preds)
                _acc_before = st.session_state.al_accuracy_before

                st.subheader("Model Comparison — Test Set Accuracy")
                st.caption(
                    f"Evaluated on the **{len(st.session_state.test_posts)}** examples "
                    "in the Testing & Evaluation tab."
                )

                _b1, _b2, _b3 = st.columns(3)
                _b1.metric("Rule-Based", f"{_acc_rule:.1%}")
                _b2.metric(
                    "ML — Before Active Learning",
                    f"{_acc_before:.1%}" if _acc_before is not None else "N/A",
                )
                _ml_delta = f"{(_acc_after - _acc_before):+.1%}" if _acc_before is not None else None
                _b3.metric("ML — After Active Learning", f"{_acc_after:.1%}", delta=_ml_delta)

                # Per-example comparison table
                st.subheader("Test Set Predictions")
                _cmp_rows = []
                for _post, _true, _rule_pred, _ml_pred in zip(
                    st.session_state.test_posts,
                    st.session_state.test_labels,
                    _rule_preds,
                    _after_preds,
                ):
                    _cmp_rows.append({
                        "Post": _post,
                        "True Label": _true,
                        "Rule-Based": _rule_pred,
                        "Rule ✓": "✓" if _rule_pred == _true else "✗",
                        "ML Model": _ml_pred,
                        "ML ✓": "✓" if _ml_pred == _true else "✗",
                    })
                st.dataframe(pd.DataFrame(_cmp_rows), use_container_width=True, hide_index=True)

            else:
                st.success("Model retrained successfully.")

            # ── Baseline comparison card ───────────────────────────────────
            if st.session_state.al_baseline_ml_acc is not None:
                st.divider()
                st.markdown("#### Baseline Summary (after Cold Start)")
                _b1, _b2 = st.columns(2)
                _b1.metric("Rule-Based Baseline",
                           f"{st.session_state.al_baseline_rule_acc:.0%}" if st.session_state.al_baseline_rule_acc is not None else "No test set",
                           help="Accuracy of the rule-based system on your test set at the time the first ML model was built.")
                _b2.metric("ML Baseline",
                           f"{st.session_state.al_baseline_ml_acc:.0%}",
                           help="Accuracy of the ML model after the cold start round.")
                st.caption("These baselines are frozen. Future rounds show improvement relative to these numbers.")

            # ── Optional 80/20 quality check ─────────────────────────────
            with st.expander("80/20 Quality Check (optional)"):
                st.caption(
                    "Randomly splits your full labeled training set 80/20 and reports accuracy "
                    "on the held-out 20%. Useful as a rough sanity check on model quality."
                )
                if st.button("Run 80/20 Quality Check"):
                    with st.spinner("Running split…"):
                        try:
                            _qc = train_with_split(ds.SAMPLE_POSTS, ds.TRUE_LABELS)
                            if _qc["warning"]:
                                st.warning(_qc["warning"])
                            _q1, _q2, _q3 = st.columns(3)
                            _q1.metric("Test Accuracy", f"{_qc['accuracy']:.1%}")
                            _q2.metric("Train Size", _qc["train_size"])
                            _q3.metric("Test Size", _qc["test_size"])
                            _qc_rows = []
                            for _cls in ["positive", "negative", "neutral", "mixed"]:
                                if _cls in _qc["report"]:
                                    _r = _qc["report"][_cls]
                                    _qc_rows.append({
                                        "Class": _cls,
                                        "Precision": f"{_r['precision']:.2f}",
                                        "Recall": f"{_r['recall']:.2f}",
                                        "F1": f"{_r['f1-score']:.2f}",
                                        "Support": int(_r["support"]),
                                    })
                            if _qc_rows:
                                st.dataframe(
                                    pd.DataFrame(_qc_rows),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                        except Exception as _e:
                            st.error(str(_e))

        st.divider()
        if st.button("Start Over (new batch)"):
            _al_reset()
            st.rerun()
