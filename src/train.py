import argparse
import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def train_model(test_size=0.2, random_state=42):
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Create output folder if missing
    os.makedirs("outputs", exist_ok=True)

    # ----- Decision Tree -----
    tree_clf = DecisionTreeClassifier(random_state=random_state)
    tree_clf.fit(X_train, y_train)
    tree_preds = tree_clf.predict(X_test)
    tree_acc = accuracy_score(y_test, tree_preds)
    print(f"Decision Tree Accuracy: {tree_acc:.4f}")

    tree_cm = confusion_matrix(y_test, tree_preds)
    tree_disp = ConfusionMatrixDisplay(confusion_matrix=tree_cm, display_labels=iris.target_names)
    tree_disp.plot(cmap="Blues", values_format="d")
    plt.title("Decision Tree - Confusion Matrix")
    plt.savefig("outputs/tree_confusion_matrix.png")
    plt.close()

    # ----- Support Vector Classifier -----
    svc_clf = SVC()
    svc_clf.fit(X_train, y_train)
    svc_preds = svc_clf.predict(X_test)
    svc_acc = accuracy_score(y_test, svc_preds)
    print(f"SVC Accuracy: {svc_acc:.4f}")

    svc_cm = confusion_matrix(y_test, svc_preds)
    svc_disp = ConfusionMatrixDisplay(confusion_matrix=svc_cm, display_labels=iris.target_names)
    svc_disp.plot(cmap="Purples", values_format="d")
    plt.title("SVC - Confusion Matrix")
    plt.savefig("outputs/svc_confusion_matrix.png")
    plt.close()

    # Return tree accuracy (for testing)
    return tree_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Decision Tree and SVC on Iris Dataset.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of test data (e.g., 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")

    args = parser.parse_args()

    train_model(test_size=args.test_size, random_state=args.random_state)
