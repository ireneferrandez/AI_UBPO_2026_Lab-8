from pathlib import Path
from PIL import Image
import joblib
import matplotlib.pyplot as plt
from src.vision.feature_extractor import extract_features

MODEL_PATH = Path("models/image_model.joblib")
MODELS_DIR = Path("models")

def load_model():
    if not MODEL_PATH.exists():
        print(f"Error: model file not found: {MODEL_PATH}")
        raise SystemExit(1)

    model = joblib.load(MODEL_PATH)
    print("Model loaded.")
    return model

def predict_image(model, image_path):
    path = Path(image_path)

    if not path.exists():
        print(f"Error: file not found: {image_path}")
        return

    with Image.open(path) as image:
        features = extract_features(image)
        image_for_plot = image.copy()

    prediction = model.predict([features])[0]

    print("=== Prediction ===")
    print(f"Image: {image_path}")
    print(f"Predicted class: {prediction}")

    plt.imshow(image_for_plot)
    plt.title(f"Prediction: {prediction}")
    plt.axis("off")
    plt.show()

def predict_with_all_models(image_path):
    path = Path(image_path)
    if not path.exists():
        print(f"Error: file not found: {image_path}")
        return

    model_files = sorted(list(MODELS_DIR.glob("*.joblib")))
    
    with Image.open(path) as image:
        features = extract_features(image)
        image_for_plot = image.copy()
        
        predictions_text = []
        print(f"=== Task 11.5: Predictions for {image_path} ===")
        
        for m_file in model_files:
            m_name = m_file.stem.replace('_', ' ').title()
            m_loaded = joblib.load(m_file)
            pred = m_loaded.predict([features])[0]
            predictions_text.append(f"{m_name}: {pred}")
            print(f"{m_name}: {pred}")

        full_title = " | ".join(predictions_text)
        plt.imshow(image_for_plot)
        plt.title(full_title, fontsize=8)
        plt.axis("off")
        plt.show()

def main():
    image_path = "data/inference_samples/noise.jpg"
    
    if MODELS_DIR.exists() and len(list(MODELS_DIR.glob("*.joblib"))) > 1:
        predict_with_all_models(image_path)
    else:
        model = load_model()
        predict_image(model, image_path)

if __name__ == "__main__":
    main()
