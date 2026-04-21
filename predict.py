from joblib import load

MODEL_PATH = "model/netflix_classifier.pkl"


def load_model(path: str = MODEL_PATH):
    """Load and return the serialized classifier."""
    return load(path)


def predict_text(text: str, path: str = MODEL_PATH) -> str:
    """Predict whether the given text belongs to a Movie or TV Show."""
    model = load_model(path)
    return str(model.predict([text])[0])
