import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "experiment_results.json")


def _standardize(train_raw, test_raw):
    """Стандартизация признаков по статистикам обучающей выборки."""
    train_matrix = np.asarray(train_raw, dtype=float)
    test_matrix = np.asarray(test_raw, dtype=float)
    column_mean = train_matrix.mean(axis=0)
    column_std = train_matrix.std(axis=0)
    column_std[column_std == 0] = 1.0
    return (train_matrix - column_mean) / column_std, (test_matrix - column_mean) / column_std


def _sigmoid(scores):
    return 1.0 / (1.0 + np.exp(-np.clip(scores, -500, 500)))


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    train_matrix, test_matrix = _standardize(data.train_features, data.test_features)
    train_target = np.asarray(data.train_target, dtype=float)
    sample_count, feature_count = train_matrix.shape

    weights = np.zeros(feature_count)
    bias = 0.0
    loss_history: list[float] = []

    for _ in range(data.epochs):
        predicted = train_matrix @ weights + bias
        residual = predicted - train_target
        loss_history.append(float(np.mean(residual ** 2)))
        weight_gradient = (2 / sample_count) * (train_matrix.T @ residual)
        bias_gradient = (2 / sample_count) * np.sum(residual)
        weights -= data.learning_rate * weight_gradient
        bias -= data.learning_rate * bias_gradient

    predictions = test_matrix @ weights + bias
    return RegressionResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        predictions=predictions,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    train_matrix, test_matrix = _standardize(data.train_features, data.test_features)
    train_target = np.asarray(data.train_target, dtype=float)
    sample_count, feature_count = train_matrix.shape

    weights = np.zeros(feature_count)
    bias = 0.0
    loss_history: list[float] = []
    epsilon = 1e-15

    for _ in range(data.epochs):
        probability = _sigmoid(train_matrix @ weights + bias)
        loss = -np.mean(
            train_target * np.log(probability + epsilon)
            + (1 - train_target) * np.log(1 - probability + epsilon)
        )
        loss_history.append(float(loss))
        residual = probability - train_target
        weight_gradient = (1 / sample_count) * (train_matrix.T @ residual)
        bias_gradient = (1 / sample_count) * np.sum(residual)
        weights -= data.learning_rate * weight_gradient
        bias -= data.learning_rate * bias_gradient

    probabilities = _sigmoid(test_matrix @ weights + bias)
    predictions = (probabilities >= 0.5).astype(int)
    return ClassificationResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        probabilities=probabilities,
        predictions=predictions,
    )


def _evaluate(true_labels, predicted_labels, predicted_scores) -> ClassificationMetrics:
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(true_labels, predicted_scores)),
        accuracy=float(accuracy_score(true_labels, predicted_labels)),
        precision=float(precision_score(true_labels, predicted_labels, zero_division=0)),
        recall=float(recall_score(true_labels, predicted_labels, zero_division=0)),
        f1=float(f1_score(true_labels, predicted_labels, zero_division=0)),
    )


def _metrics_as_dict(metrics: ClassificationMetrics) -> dict[str, float]:
    return {
        "roc_auc": metrics.roc_auc,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
    }


def _store_classification_results(own, logistic, tree, knn) -> None:
    results: dict = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as results_file:
            results = json.load(results_file)
    results["classification"] = {
        "own_logistic": _metrics_as_dict(own),
        "logistic_regression": _metrics_as_dict(logistic),
        "decision_tree": _metrics_as_dict(tree),
        "knn": _metrics_as_dict(knn),
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    test_target = np.asarray(data.test_target, dtype=int)
    train_target = np.asarray(data.train_target, dtype=int)
    train_matrix, test_matrix = _standardize(data.train_features, data.test_features)

    own_model = train_logistic_classifier(data)
    own_metrics = _evaluate(test_target, own_model.predictions, own_model.probabilities)

    logistic_model = SklearnLogisticRegression(max_iter=1000)
    logistic_model.fit(train_matrix, train_target)
    logistic_metrics = _evaluate(
        test_target,
        logistic_model.predict(test_matrix),
        logistic_model.predict_proba(test_matrix)[:, 1],
    )

    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(train_matrix, train_target)
    tree_metrics = _evaluate(
        test_target,
        tree_model.predict(test_matrix),
        tree_model.predict_proba(test_matrix)[:, 1],
    )

    knn_model = KNeighborsClassifier()
    knn_model.fit(train_matrix, train_target)
    knn_metrics = _evaluate(
        test_target,
        knn_model.predict(test_matrix),
        knn_model.predict_proba(test_matrix)[:, 1],
    )

    _store_classification_results(own_metrics, logistic_metrics, tree_metrics, knn_metrics)

    return BaselineComparisonResult(
        logistic_regression=logistic_metrics,
        decision_tree=tree_metrics,
        knn=knn_metrics,
    )


def _run_on_real_data() -> None:
    """Прогон на реальных датасетах и запись experiment_results.json."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    student = pd.read_csv("Student_Performance.csv")
    student["Extracurricular Activities"] = student["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    reg_target = student.pop("Performance Index").to_numpy(dtype=float)
    reg_features = student.to_numpy(dtype=float)
    reg_train_x, reg_test_x, reg_train_y, reg_test_y = train_test_split(
        reg_features, reg_target, test_size=0.2, random_state=42
    )
    regression = train_linear_regression(
        RegressionInput(reg_train_x, reg_train_y, reg_test_x, reg_test_y)
    )
    test_mse = float(np.mean((regression.predictions - reg_test_y) ** 2))

    german = pd.read_csv("german.csv", sep=";")
    cls_target = german.pop("Creditability").to_numpy(dtype=int)
    cls_features = german.to_numpy(dtype=float)
    cls_train_x, cls_test_x, cls_train_y, cls_test_y = train_test_split(
        cls_features, cls_target, test_size=0.2, random_state=42, stratify=cls_target
    )
    compare_classical_models(
        ClassificationInput(cls_train_x, cls_train_y, cls_test_x, cls_test_y)
    )

    with open(RESULTS_PATH, encoding="utf-8") as results_file:
        results = json.load(results_file)
    results["linear_regression"] = {"test_mse": test_mse}
    with open(RESULTS_PATH, "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)
    print(f"test MSE = {test_mse:.4f}; результаты в {RESULTS_PATH}")


if __name__ == "__main__":
    _run_on_real_data()
