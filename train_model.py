import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

DATA_PATH = Path("data/diabetes.csv")
MODEL_PATH = Path("model/diabetes_model.pkl")
METRICS_PATH = Path("model/metrics.json")

def main():
    df = pd.read_csv(DATA_PATH)
    target_col = "Outcome" if "Outcome" in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    numeric_features = X.columns.tolist()
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    preprocess = ColumnTransformer([("num", num_pipe, numeric_features)], remainder="drop")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=42),
        "SVM-RBF": SVC(kernel="rbf", probability=True, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, clf in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("clf", clf)])
        auc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        results[name] = {
            "roc_auc_mean": float(np.mean(auc_scores)),
            "roc_auc_std": float(np.std(auc_scores)),
        }

    best_name = max(results.keys(), key=lambda k: results[k]["roc_auc_mean"])
    best_clf = models[best_name]
    best_pipe = Pipeline([("preprocess", preprocess), ("clf", best_clf)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "cv_results": results,
        "best_model": best_name,
        "holdout_accuracy": float(accuracy_score(y_test, y_pred)),
        "holdout_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "features": numeric_features,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_pipe, f)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print("Best model:", best_name)
    print("Saved model to:", MODEL_PATH)
    print("Saved metrics to:", METRICS_PATH)

if __name__ == "__main__":
    main()
