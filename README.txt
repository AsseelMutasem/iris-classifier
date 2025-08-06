# Iris Classifier (Decision Tree & SVC)

## Overview

End-to-end ML example from the AI Fundamentals course – this project builds and compares a **Decision Tree** and an **SVC (Support Vector Classifier)** on the classic Iris dataset using scikit-learn.
It includes a reproducible Python script, visual output of confusion matrices, and test coverage.

## Quick start

```bash
git clone https://github.com/AsseelMutasem/iris-classifier.git
cd iris-classifier
python -m venv venv && .\venv\Scripts\activate
pip install -r requirements.txt
python src/train.py --test-size 0.2 --random-state 42
Of course! It looks like the project structure tree in your README was not formatted correctly, making it hard to read.

Here is the corrected version with the proper tree structure and the fixed command in the "Quick start" section.

```markdown
# Iris Classifier (Decision Tree & SVC)

## Overview

End-to-end ML example from the AI Fundamentals course – this project builds and compares a **Decision Tree** and an **SVC (Support Vector Classifier)** on the classic Iris dataset using scikit-learn.
It includes a reproducible Python script, visual output of confusion matrices, and test coverage.

## Quick start

```bash
git clone https://github.com/AsseelMutasem/iris-classifier.git
cd iris-classifier
python -m venv venv && .\venv\Scripts\activate
pip install -r requirements.txt
python src/train.py --test-size 0.2 --random-state 42
```

## Project Structure

```
iris-classifier/
├── data/                    # empty (data is loaded from sklearn)
├── notebooks/
│   └── iris_model.ipynb     # exploratory notebook
├── src/
│   └── train.py             # CLI script to train models
├── tests/
│   └── test_train.py        # accuracy test for both models
├── outputs/
│   ├── tree_confusion_matrix.png
│   └── svc_confusion_matrix.png
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```
