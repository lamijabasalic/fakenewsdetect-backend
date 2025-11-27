from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import joblib
import os
import json
import pandas as pd
import subprocess
try:
    from preprocessing import preprocess_bosnian_text
except ImportError:
    from app.preprocessing import preprocess_bosnian_text

app = FastAPI(
    title="Bosnian Fake News Detector API",
    description="API za detekciju lažnih vijesti na bosanskom jeziku.",
    version="2.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")
# Dataset is now in backend/dataset/ folder
BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_PATH = os.path.join(BACKEND_DIR, "dataset", "bosnian_fake_news_dataset.csv")

VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "classifier.joblib")
METRICS_PATH = os.path.join(METRICS_DIR, "results.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

class NewsInput(BaseModel):
    title: str
    text: str

class NewsItem(BaseModel):
    title: str
    text: str
    label: int  # 0 = Real, 1 = Fake

class ClassificationResponse(BaseModel):
    label: str  # "FAKE" or "REAL"
    probability: float
    explanation: str
    confidence: float

class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    train_size: int
    test_size: int
    total_samples: int
    model_type: str

def load_artifacts():
    """Load model and vectorizer from disk."""
    if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH)):
        return None, None
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
        return vectorizer, model
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return None, None

vectorizer, model = load_artifacts()

def get_top_tokens(text: str, vectorizer, model, top_n: int = 5) -> List[str]:
    """Extract top contributing tokens for explanation."""
    try:
        processed = preprocess_bosnian_text(text, use_stemming=False)
        X = vectorizer.transform([processed])
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get coefficients for fake class (assuming class 1 is fake)
        if hasattr(model, "coef_"):
            coef = model.coef_[0]
            # Get indices of non-zero features
            indices = X.indices
            # Get corresponding coefficients
            feature_coefs = [(feature_names[i], coef[i]) for i in indices]
            # Sort by absolute coefficient value
            feature_coefs.sort(key=lambda x: abs(x[1]), reverse=True)
            # Return top N tokens
            return [token for token, _ in feature_coefs[:top_n]]
    except:
        pass
    return []

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Bosnian Fake News Detector API radi.",
        "version": "2.0.0",
        "endpoints": {
            "/classify": "POST - Classify news article",
            "/train": "POST - Retrain model",
            "/metrics": "GET - Get evaluation metrics",
            "/dataset": "GET - View dataset, POST - Add new item"
        }
    }

@app.post("/classify", response_model=ClassificationResponse)
def classify_news(data: NewsInput):
    """
    Classify news article as FAKE or REAL.
    Returns label, probability, confidence, and explanation.
    """
    global vectorizer, model

    if vectorizer is None or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model nije treniran. Molimo pokrenite /train endpoint prvo."
        )

    # Preprocess text
    text_combined = data.title + " " + data.text
    processed_text = preprocess_bosnian_text(text_combined, use_stemming=False)
    
    # Vectorize
    X = vectorizer.transform([processed_text])
    
    # Predict
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)[0]
        # Assuming class 1 is FAKE, class 0 is REAL
        fake_prob = float(probas[1])
        real_prob = float(probas[0])
        label = int(model.classes_[probas.argmax()])
        confidence = max(fake_prob, real_prob)
    else:
        label = int(model.predict(X)[0])
        fake_prob = 0.5 if label == 1 else 0.5
        confidence = 0.5

    is_fake = label == 1
    
    # Get explanation tokens
    explanation_tokens = get_top_tokens(text_combined, vectorizer, model, top_n=5)
    explanation = f"Model je analizirao tekst i identificirao ključne riječi: {', '.join(explanation_tokens[:3]) if explanation_tokens else 'N/A'}"

    return ClassificationResponse(
        label="FAKE" if is_fake else "REAL",
        probability=fake_prob,
        explanation=explanation,
        confidence=confidence
    )

@app.post("/predict", response_model=ClassificationResponse)
def predict_news(data: NewsInput):
    """Legacy endpoint - redirects to /classify."""
    return classify_news(data)

@app.post("/train")
def train_model():
    """
    Retrain the model on the current dataset.
    This will update the model and metrics.
    """
    try:
        # Run training script
        script_path = os.path.join(os.path.dirname(__file__), "train_model.py")
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(script_path)
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Greška pri treniranju: {result.stderr}"
            )
        
        # Reload artifacts
        global vectorizer, model
        vectorizer, model = load_artifacts()
        
        return {
            "message": "Model uspješno treniran.",
            "output": result.stdout
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Greška pri treniranju modela: {str(e)}"
        )

@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """Get evaluation metrics from the last training."""
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(
            status_code=404,
            detail="Metrike nisu dostupne. Molimo trenirajte model prvo."
        )
    
    try:
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        return MetricsResponse(**metrics)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Greška pri učitavanju metrika: {str(e)}"
        )

@app.get("/dataset")
def get_dataset():
    """Get all items from the dataset."""
    try:
        df = pd.read_csv(DATASET_PATH)
        # Convert to list of dicts
        items = df.to_dict('records')
        return {
            "total": len(items),
            "items": items
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Greška pri učitavanju dataset-a: {str(e)}"
        )

@app.post("/dataset")
def add_dataset_item(item: NewsItem):
    """Add a new item to the dataset."""
    try:
        # Read existing dataset
        df = pd.read_csv(DATASET_PATH)
        
        # Add new item
        new_id = df['id'].max() + 1 if len(df) > 0 else 1
        new_row = {
            'id': new_id,
            'title': item.title,
            'text': item.text,
            'label': item.label
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save back to CSV
        df.to_csv(DATASET_PATH, index=False)
        
        return {
            "message": "Item uspješno dodan u dataset.",
            "id": int(new_id),
            "total_items": len(df)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Greška pri dodavanju itema: {str(e)}"
        )
