from joblib import load

MODEL_PATH = "model/netflix_classifier.pkl"


def predict_content_type(text: str) -> str:
    """Load the trained model and predict whether content is a Movie or TV Show."""
    model = load(MODEL_PATH)
    prediction = model.predict([text])[0]
    return str(prediction)


if __name__ == "__main__":
    print("Netflix Content Classifier")
    print("Type a description and press Enter. Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter title description: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not user_input:
            print("Please enter a valid description.\n")
            continue

        result = predict_content_type(user_input)
        print(f"Predicted class: {result}\n")
