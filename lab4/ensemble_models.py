"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

MODEL_NAME = "SoftVoting(LogReg+RandomForest+GradientBoosting)"


def _build_ensemble(random_seed: int) -> VotingClassifier:
    """Собирает soft-voting ансамбль из линейной модели и двух древесных."""
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=random_seed),
    )
    forest = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_seed,
    )
    boosting = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=random_seed,
    )
    return VotingClassifier(
        estimators=[("logistic", logistic), ("forest", forest), ("boosting", boosting)],
        voting="soft",
        weights=[2, 1, 2],
    )


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """
    train_features = np.asarray(data.train_features, dtype=float)
    train_target = np.asarray(data.train_target, dtype=int)
    test_features = np.asarray(data.test_features, dtype=float)
    test_target = np.asarray(data.test_target, dtype=int)

    model = _build_ensemble(data.random_seed)
    model.fit(train_features, train_target)

    probabilities = model.predict_proba(test_features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return EnsembleResult(
        model_name=MODEL_NAME,
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(test_target, probabilities)),
        accuracy=float(accuracy_score(test_target, predictions)),
        precision=float(precision_score(test_target, predictions, zero_division=0)),
        recall=float(recall_score(test_target, predictions, zero_division=0)),
        f1=float(f1_score(test_target, predictions, zero_division=0)),
    )


def _run_on_real_data() -> None:
    """Прогон экспериментов на german.csv и запись experiment_results.json."""
    import json
    import os

    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    frame = pd.read_csv("german.csv", sep=";")
    target = frame.pop("Creditability").to_numpy(dtype=int)
    features = frame.to_numpy(dtype=float)
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    def scored(model) -> dict[str, float]:
        model.fit(train_x, train_y)
        scores = model.predict_proba(test_x)[:, 1]
        labels = (scores >= 0.5).astype(int)
        return {
            "roc_auc": float(roc_auc_score(test_y, scores)),
            "accuracy": float(accuracy_score(test_y, labels)),
            "precision": float(precision_score(test_y, labels, zero_division=0)),
            "recall": float(recall_score(test_y, labels, zero_division=0)),
            "f1": float(f1_score(test_y, labels, zero_division=0)),
        }

    experiments = {
        "RandomForest": scored(
            RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=42)
        ),
        "GradientBoosting": scored(
            GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
        ),
        "HistGradientBoosting": scored(
            HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, random_state=42)
        ),
    }

    selected = run_ensemble_experiment(EnsembleInput(train_x, train_y, test_x, test_y, 42))
    selected_metrics = {
        "roc_auc": selected.roc_auc,
        "accuracy": selected.accuracy,
        "precision": selected.precision,
        "recall": selected.recall,
        "f1": selected.f1,
    }
    experiments[selected.model_name] = selected_metrics

    results = {
        "selected_model": selected.model_name,
        "selected_metrics": selected_metrics,
        "experiments": experiments,
    }
    path = os.path.join(os.path.dirname(__file__), "experiment_results.json")
    with open(path, "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)
    print(f"ROC-AUC = {selected.roc_auc:.4f}; результаты в {path}")


if __name__ == "__main__":
    _run_on_real_data()
