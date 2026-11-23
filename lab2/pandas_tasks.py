"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    frame = pd.read_csv(data.csv_path)

    missing_by_column = {column: int(frame[column].isna().sum()) for column in frame.columns}
    adults_over_30_count = int((frame["Age"] > 30).sum())

    mean_age = frame.groupby("Pclass")["Age"].mean()
    mean_age_by_pclass = {int(pclass): float(value) for pclass, value in mean_age.items()}

    survival = frame.groupby("Pclass")["Survived"].mean()
    survival_rate_by_pclass = {int(pclass): float(value) for pclass, value in survival.items()}

    highest_fares = [float(fare) for fare in frame["Fare"].sort_values(ascending=False).head(5)]

    return TitanicSummary(
        row_count=int(len(frame)),
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )
