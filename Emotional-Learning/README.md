# Mood Machine

A mood classifier that compares a hand-crafted rule-based model against a machine learning model to classify text as **positive**, **negative**, or **mixed**.

See [model_card.md](model_card.md) for a full breakdown of how the models work, evaluation results, and limitations.

---

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the app

```bash
streamlit run frontend/app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Project structure

```
Emotional-Learning/
├── backend/
│   ├── mood_analyzer.py    # Rule-based classifier
│   ├── ml_model.py         # ML classifier (Bag of Words + scikit-learn)
│   ├── active_learner.py   # Active learning loop
│   └── dataset.py          # Sample data
├── frontend/
│   └── app.py              # Streamlit UI
├── requirements.txt
└── model_card.md
```
