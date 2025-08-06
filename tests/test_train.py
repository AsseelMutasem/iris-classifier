from src.train import train_model

def test_both_models_accuracy():
    tree_acc, svc_acc = train_model(test_size=0.2, random_state=42)
    assert tree_acc >= 0.9, f"Decision Tree accuracy too low: {tree_acc}"
    assert svc_acc >= 0.9, f"SVC accuracy too low: {svc_acc}"
