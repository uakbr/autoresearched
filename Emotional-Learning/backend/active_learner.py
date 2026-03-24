"""
active_learner.py

Backend logic for the Active Learning tab:
  - Vectorize unlabeled texts with TF-IDF for clustering
  - Cluster with K-means or DBSCAN
  - Sample representative examples per cluster for human labeling
  - Retrain the ML model with an 80/20 train/test split and return metrics
"""

from typing import List, Dict, Tuple, Optional
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from ml_model import train_ml_model, predict_single_text


# ── Vectorization ─────────────────────────────────────────────────────────────

def vectorize_texts(texts: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    """
    Fit a TF-IDF vectorizer on the given texts and return the
    fitted vectorizer and the dense feature matrix.

    Uses unigrams + bigrams and sublinear TF scaling to produce
    more meaningful distances for short social-media style posts.
    """
    vec = TfidfVectorizer(
        max_features=500,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts).toarray()
    return vec, X


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_texts(
    X: np.ndarray,
    algorithm: str,
    k: int = 4,
    eps: float = 0.5,
    min_samples: int = 2,
) -> np.ndarray:
    """
    Cluster the TF-IDF matrix X.

    algorithm: "kmeans" or "dbscan"

    K-means kwargs: k (number of clusters)
    DBSCAN kwargs:  eps (neighbourhood radius), min_samples

    Returns an array of cluster labels (ints). DBSCAN uses -1 for noise.
    """
    if algorithm == "kmeans":
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        return model.fit_predict(X)
    elif algorithm == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        return model.fit_predict(X)
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose 'kmeans' or 'dbscan'.")


def compute_elbow_data(
    X: np.ndarray,
    max_k: int = 10,
) -> List[Tuple[int, float]]:
    """
    Run K-means for k = 2..max_k and return (k, inertia) pairs.
    Used to render the elbow plot in the UI.
    """
    results = []
    for k in range(2, max_k + 1):
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X)
        results.append((k, float(km.inertia_)))
    return results


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_from_clusters(
    texts: List[str],
    cluster_labels: np.ndarray,
    X: np.ndarray,
    algorithm: str,
    n_per_cluster: int,
) -> Dict[int, List[int]]:
    """
    For each cluster, pick the n_per_cluster most representative examples.

    K-means: picks examples closest to the cluster centroid.
    DBSCAN:  random sample (no centroid exists).

    Returns: { cluster_id: [text_index, ...] }
    """
    unique_clusters = sorted(set(cluster_labels.tolist()))
    samples: Dict[int, List[int]] = {}

    for cid in unique_clusters:
        member_indices = [i for i, lbl in enumerate(cluster_labels) if lbl == cid]
        if not member_indices:
            continue

        if algorithm == "kmeans" and cid != -1:
            # centroid of this cluster
            member_vecs = X[member_indices]
            centroid = member_vecs.mean(axis=0)
            dists = np.linalg.norm(member_vecs - centroid, axis=1)
            sorted_by_dist = [member_indices[i] for i in np.argsort(dists)]
            samples[cid] = sorted_by_dist[:n_per_cluster]
        else:
            # DBSCAN (or noise cluster): random sample
            chosen = random.sample(member_indices, min(n_per_cluster, len(member_indices)))
            samples[cid] = chosen

    return samples


# ── Uncertainty Sampling ──────────────────────────────────────────────────────

def uncertainty_sampling(
    labeled_texts: List[str],
    labeled_labels: List[str],
    unlabeled_texts: List[str],
    n_samples: int,
) -> List[int]:
    """
    Select the unlabeled examples the current model is most uncertain about.

    Trains a TF-IDF + LogisticRegression model on the labeled data,
    runs predict_proba on all unlabeled texts, and returns the indices
    of the n_samples examples with the lowest max class probability
    (i.e. the model is least confident about those).

    Returns a list of indices into unlabeled_texts, sorted most-uncertain first.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    if len(labeled_texts) < 2:
        # Not enough data to train — fall back to random sampling
        indices = list(range(len(unlabeled_texts)))
        random.shuffle(indices)
        return indices[:n_samples]

    vec = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    X_labeled = vec.fit_transform(labeled_texts)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    try:
        clf.fit(X_labeled, labeled_labels)
    except ValueError:
        # Single class in labels — fall back to random
        indices = list(range(len(unlabeled_texts)))
        random.shuffle(indices)
        return indices[:n_samples]

    X_unlabeled = vec.transform(unlabeled_texts)
    proba = clf.predict_proba(X_unlabeled)           # shape: (n_unlabeled, n_classes)
    max_confidence = proba.max(axis=1)               # highest class probability per example
    uncertainty_order = np.argsort(max_confidence)   # ascending = most uncertain first

    return uncertainty_order[:n_samples].tolist()


# ── Retraining with 80/20 split ───────────────────────────────────────────────

def train_with_split(
    all_texts: List[str],
    all_labels: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train the ML model on an 80% split and evaluate on the held-out 20%.

    Returns a dict with:
      vectorizer, model       — the newly fitted objects (replace session state)
      train_size, test_size   — int counts
      accuracy                — float 0-1
      report                  — classification_report dict (per-class metrics)
      y_test, y_pred          — lists for confusion matrix rendering
      warning                 — str or None (e.g. stratify fallback notice)
    """
    warning = None

    # Try stratified split first; fall back if any class is too small
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            all_texts,
            all_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=all_labels,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            all_texts,
            all_labels,
            test_size=test_size,
            random_state=random_state,
        )
        warning = (
            "Some classes have very few examples — stratified split was not possible. "
            "Accuracy estimates may be noisy. Try adding more labeled examples."
        )

    vec, model = train_ml_model(X_train, y_train)

    X_test_vec = vec.transform(X_test)
    y_pred = model.predict(X_test_vec).tolist()

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    return {
        "vectorizer": vec,
        "model": model,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "report": report,
        "warning": warning,
    }
