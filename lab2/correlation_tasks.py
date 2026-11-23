"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]


def _mri_correlations(group: pd.DataFrame) -> dict[str, float]:
    return {feature: float(group[feature].corr(group["MRI_Count"])) for feature in FEATURES}


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    frame = pd.read_csv(data.csv_path, sep="\t", na_values="NA")

    male = frame[frame["Gender"] == "Male"]
    female = frame[frame["Gender"] == "Female"]

    male_corr = _mri_correlations(male)
    female_corr = _mri_correlations(female)

    strongest_feature = max(
        [*male_corr.items(), *female_corr.items()],
        key=lambda item: abs(item[1]),
    )[0]

    return BrainCorrelationSummary(
        men_count=int(len(male)),
        women_count=int(len(female)),
        women_mri_correlation=female_corr,
        men_mri_correlation=male_corr,
        strongest_mri_feature=strongest_feature,
    )


def _plot_experiments() -> None:
    """Scatter-графики признаков против MRI_Count и heatmap корреляций."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    frame = pd.read_csv("brainsize.txt", sep="\t", na_values="NA")
    frame["Пол"] = frame["Gender"].map({"Male": "Мужчины", "Female": "Женщины"})

    figure, axes = plt.subplots(1, len(FEATURES), figsize=(4 * len(FEATURES), 4))
    figure.suptitle("Зависимость объёма МРТ от признаков по полу")
    for index, (axis, feature) in enumerate(zip(axes, FEATURES)):
        sns.scatterplot(
            data=frame, x=feature, y="MRI_Count", hue="Пол", ax=axis,
            legend=(index == 0),
        )
        axis.set_title(f"{feature} и объём МРТ")
        axis.set_xlabel(f"Признак {feature}")
        axis.set_ylabel("Объём МРТ (MRI_Count)")
    figure.tight_layout()
    figure.savefig("scatter_mri.png", dpi=120)
    plt.close(figure)

    correlation = frame[FEATURES + ["MRI_Count"]].corr()
    heatmap_figure, heatmap_axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", ax=heatmap_axis)
    heatmap_axis.set_title("Матрица корреляций признаков")
    heatmap_figure.tight_layout()
    heatmap_figure.savefig("correlation_heatmap.png", dpi=120)
    print("Сохранено: scatter_mri.png, correlation_heatmap.png")


if __name__ == "__main__":
    _plot_experiments()
