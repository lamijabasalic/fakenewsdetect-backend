"""
Script za treniranje modela na bosanskom datasetu za detekciju lažnih vijesti.

Pokretanje:
    python train_model.py
"""

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib
try:
    from preprocessing import preprocess_bosnian_text
except ImportError:
    from app.preprocessing import preprocess_bosnian_text

# Dataset is now in backend/dataset/ folder
BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_PATH = os.path.join(BACKEND_DIR, "dataset", "bosnian_fake_news_dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

def main():
    print("Učitavanje dataset-a...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Učitano {len(df)} primjera.")

    # Priprema teksta: kombinujemo naslov i tekst
    print("Preprocesiranje teksta...")
    df["combined"] = (df["title"].fillna("") + " " + df["text"].fillna(""))
    
    # Primjenjujemo Bosnian preprocessing
    df["processed"] = df["combined"].apply(lambda x: preprocess_bosnian_text(x, use_stemming=False))

    X = df["processed"]
    y = df["label"]

    # Split dataset 80/20
    print("Dijeljenje dataset-a (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {len(X_train)} primjera")
    print(f"Test set: {len(X_test)} primjera")

    # TF-IDF Vectorization
    print("Vektorizacija TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Logistic Regression classifier
    print("Treniranje Logistic Regression modela...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_vec, y_train)

    # Predictions
    print("Evaluacija modela...")
    y_pred = clf.predict(X_test_vec)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", pos_label=1)

    # Prepare metrics dictionary
    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "total_samples": int(len(df)),
        "model_type": "TF-IDF + Logistic Regression"
    }

    # Print results
    print("\n" + "="*50)
    print("REZULTATI EVALUACIJE:")
    print("="*50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print("\nDetaljan izvještaj:\n")
    print(classification_report(y_test, y_pred, digits=4))

    # Save model artifacts
    vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
    model_path = os.path.join(MODEL_DIR, "classifier.joblib")
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(clf, model_path)

    print(f"\nModel spašen u: {model_path}")
    print(f"Vectorizer spašen u: {vectorizer_path}")

    # Save metrics to JSON
    metrics_path = os.path.join(METRICS_DIR, "results.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"Metrike spašene u: {metrics_path}")
    print("\nTreniranje završeno uspješno!")

if __name__ == "__main__":
    main()
