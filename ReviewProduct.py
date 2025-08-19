"""
Review NLP Mini‑System
======================
A compact, production‑ready example for processing customer product reviews.

Features
- Supervised sentiment classification (if you have labels)
- Unsupervised topic discovery via clustering (when you don't)
- REST API using FastAPI for /train and /predict
- Language‑agnostic defaults (character n‑grams handle Chinese/Japanese/English without external tokenizers)

Quick requirements (Python 3.9+ recommended)
    pip install fastapi uvicorn[standard] scikit-learn pandas joblib

Run the API
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Data format
- CSV with at least a column named 'text'.
- Optional: 'label' (e.g., pos/neg/neutral or 1/0). Strings or integers are both fine.

Example /train body (JSON)
{
  "csv_path": "./reviews.csv",
  "label_col": "label",
  "mode": "supervised",  // or "cluster"
  "clusters": 5            // only used when mode=="cluster"
}

Example /predict body (JSON)
{
  "texts": ["电池续航不错，就是有点重", "Terrible quality, broke in 2 days"]
}

"""
from __future__ import annotations
import os
import json
import joblib
import pandas as pd
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans

MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "review_sentiment.joblib")

app = FastAPI(title="Review NLP Mini‑System", version="1.0.0")

# -----------------------------
# Data utilities
# -----------------------------
def _load_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column")
    # Drop NA texts
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    return df

# -----------------------------
# Supervised sentiment pipeline
# -----------------------------
def make_pipeline() -> Pipeline:
    # Character n‑grams work well across languages without external tokenizers
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_features=100_000
        )),
        ("clf", LogisticRegression(
            max_iter=200,
            n_jobs=None,
            class_weight="balanced"
        ))
    ])


def train_supervised(csv_path: str, label_col: str = "label") -> Dict[str, Any]:
    df = _load_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"CSV must contain a '{label_col}' column for supervised training")

    X = df["text"].astype(str).tolist()
    y = df[label_col].astype(str).tolist()

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # fallback when classes are too imbalanced for stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    pipe = make_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    return {
        "saved_to": MODEL_PATH,
        "accuracy": acc,
        "report": report,
        "classes": sorted(list(set(y)))
    }

# -----------------------------
# Unsupervised topic discovery
# -----------------------------
def cluster_reviews(csv_path: str, k: int = 5) -> Dict[str, Any]:
    df = _load_csv(csv_path)
    texts = df["text"].astype(str).tolist()

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=100_000)
    X = vectorizer.fit_transform(texts)

    # Simple KMeans clustering; for larger corpora you can switch to MiniBatchKMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    # Extract top terms per cluster
    terms = vectorizer.get_feature_names_out()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    topics: Dict[str, Any] = {}
    for i in range(k):
        top_chars = [terms[ind] for ind in order_centroids[i, :20]]
        topics[str(i)] = {
            "size": int((labels == i).sum()),
            # Join contiguous char-ngrams into readable hints (rough heuristic)
            "top_ngrams": top_chars
        }

    return {
        "n_texts": len(texts),
        "n_clusters": k,
        "topics": topics,
        "cluster_labels": labels.tolist()
    }

# -----------------------------
# I/O schemas
# -----------------------------
class TrainBody(BaseModel):
    csv_path: str = Field(..., description="Path to CSV with 'text' and optional 'label' columns")
    label_col: Optional[str] = Field("label", description="Name of label column for supervised training")
    mode: str = Field("supervised", description="'supervised' for sentiment, 'cluster' for topic discovery")
    clusters: Optional[int] = Field(5, description="Number of clusters for topic discovery")

class PredictBody(BaseModel):
    texts: List[str]

# -----------------------------
# API endpoints
# -----------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    status = "ready" if os.path.exists(MODEL_PATH) else "no-model"
    return {"status": status}

@app.post("/train")
def train_endpoint(body: TrainBody) -> Dict[str, Any]:
    if body.mode not in {"supervised", "cluster"}:
        raise HTTPException(status_code=400, detail="mode must be 'supervised' or 'cluster'")

    if body.mode == "supervised":
        try:
            result = train_supervised(body.csv_path, body.label_col or "label")
            return {"ok": True, "mode": "supervised", "result": result}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        try:
            k = int(body.clusters or 5)
            result = cluster_reviews(body.csv_path, k)
            return {"ok": True, "mode": "cluster", "result": result}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict_endpoint(body: PredictBody) -> Dict[str, Any]:
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="No trained model found. Call /train in supervised mode first.")

    pipe: Pipeline = joblib.load(MODEL_PATH)
    probs_ok = hasattr(pipe.named_steps.get("clf"), "predict_proba")

    texts = [str(t) for t in body.texts]
    preds = pipe.predict(texts)
    resp: Dict[str, Any] = {"predictions": preds.tolist() if hasattr(preds, "tolist") else list(preds)}

    if probs_ok:
        proba = pipe.predict_proba(texts)
        classes = list(pipe.named_steps["clf"].classes_)
        resp["probabilities"] = [dict(zip(classes, row.tolist())) for row in proba]

    return resp

# -----------------------------
# Local CLI helper (optional)
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Review NLP Mini‑System")
    parser.add_argument("csv", help="Path to CSV with 'text' and optional 'label'")
    parser.add_argument("--mode", choices=["supervised", "cluster"], default="supervised")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--clusters", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "supervised":
        out = train_supervised(args.csv, args.label_col)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        out = cluster_reviews(args.csv, args.clusters)
        print(json.dumps(out, ensure_ascii=False, indent=2))
