import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from src.vision.feature_extractor import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

DATASET_DIR = Path("data/processed/images")

def load_image_split(split_dir):
    X = []
    y = []

    class_dirs = sorted([
        path for path in split_dir.iterdir()
        if path.is_dir()
    ])
    
    for class_dir in class_dirs:
        class_name = class_dir.name

        image_files = sorted([
            path for path in class_dir.iterdir()
            if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        for image_path in image_files:
            with Image.open(image_path) as image:
                features = extract_features(image)
                X.append(features)
                y.append(class_name)

    X = np.array(X)
    y = np.array(y)

    return X, y

def load_training_and_test_data():
    train_dir = DATASET_DIR / "train"
    test_dir = DATASET_DIR / "test"
    
    X_train, y_train = load_image_split(train_dir)
    X_test, y_test = load_image_split(test_dir)

    print("=== Image ML Dataset ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

def train_and_compare_models(X_train, X_test, y_train, y_test):
    print("=== Task 11: Comparing Multiple Models ===")
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC()
    }

    results = []

    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results.append({
            "name": name,
            "model": model,
            "accuracy": accuracy,
            "time": training_time
        })

        print(f"Model: {name} | Accuracy: {accuracy:.4f} | Time: {training_time:.4f}s")
    
    return results

def plot_comparison(results):
    names = [r["name"] for r in results]
    accs = [r["accuracy"] for r in results]
    times = [r["time"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.scatter(times, accs, color='blue', s=100)

    for i, name in enumerate(names):
        plt.annotate(name, (times[i], accs[i]), xytext=(5, 5), textcoords='offset points')

    plt.xlabel("Training Time (s)")
    plt.ylabel("Accuracy")
    plt.title("Task 11: Accuracy vs Training Time")
    plt.grid(True)
    
    Path("reports").mkdir(exist_ok=True)
    plt.savefig("reports/model_comparison.png")
    plt.show()

def save_all_models(results):
    Path("models").mkdir(parents=True, exist_ok=True)
    for r in results:
        path = Path(f"models/{r['name'].lower().replace(' ', '_')}.joblib")
        joblib.dump(r['model'], path)

def main():
    X_train, X_test, y_train, y_test = load_training_and_test_data()

    results = train_and_compare_models(X_train, X_test, y_train, y_test)

    plot_comparison(results)

    save_all_models(results)

if __name__ == "__main__":
    main()
