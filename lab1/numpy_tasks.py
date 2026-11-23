"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    size = np.asarray(matrices[0]).shape[0]
    accumulator = np.zeros((size, 1))
    for matrix, vector in zip(matrices, vectors):
        accumulator = accumulator + np.asarray(matrix) @ np.asarray(vector).reshape(size, 1)
    return accumulator


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    source = np.asarray(matrix)
    return np.where(source > threshold, 1, 0)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [list(np.unique(row)) for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [list(np.unique(matrix[:, col])) for col in range(matrix.shape[1])]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    generator = np.random.default_rng(seed)
    sample = generator.normal(loc=mean, scale=std, size=(rows, columns))
    return MatrixStatistics(
        matrix=sample,
        row_means=sample.mean(axis=1),
        column_means=sample.mean(axis=0),
        row_variances=sample.var(axis=1),
        column_variances=sample.var(axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    board = np.full((rows, columns), second)
    board[::2, ::2] = first
    board[1::2, 1::2] = first
    return board


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    canvas = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)
    top = (image_height - height) // 2
    left = (image_width - width) // 2
    canvas[top:top + height, left:left + width] = shape_color
    return canvas


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    canvas = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)
    center_y, center_x = image_height // 2, image_width // 2
    ys, xs = np.ogrid[:image_height, :image_width]
    inside = ((xs - center_x) ** 2) / semi_axis_x ** 2 + ((ys - center_y) ** 2) / semi_axis_y ** 2 <= 1
    canvas[inside] = shape_color
    return canvas


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = np.asarray(data.values, dtype=float), data.window
    maxima = [i for i in range(1, len(values) - 1) if values[i] > values[i - 1] and values[i] > values[i + 1]]
    minima = [i for i in range(1, len(values) - 1) if values[i] < values[i - 1] and values[i] < values[i + 1]]
    kernel = np.ones(window) / window
    moving = np.convolve(values, kernel, mode="valid")
    return TimeSeriesStatistics(
        mean=float(values.mean()),
        variance=float(values.var()),
        std=float(values.std()),
        local_maxima_indices=np.array(maxima, dtype=int),
        local_minima_indices=np.array(minima, dtype=int),
        moving_average=moving,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = np.asarray(data.labels, dtype=int), data.class_count
    width = class_count if class_count is not None else int(labels.max()) + 1
    encoded = np.zeros((len(labels), width), dtype=int)
    encoded[np.arange(len(labels)), labels] = 1
    return encoded


def _plot_histograms() -> None:
    """Гистограммы значений по первым строкам и столбцам случайной матрицы (задача 4)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = matrix_statistics(RandomMatrixInput(rows=200, columns=200, seed=42))
    matrix = stats.matrix
    count = 4
    figure, axes = plt.subplots(2, count, figsize=(4 * count, 7))
    figure.suptitle("Гистограммы значений матрицы 200×200")
    for index in range(count):
        axes[0, index].hist(matrix[index], bins=20, color="steelblue")
        axes[0, index].set_title(
            f"Строка {index}: μ={stats.row_means[index]:.2f}, σ²={stats.row_variances[index]:.2f}"
        )
        axes[1, index].hist(matrix[:, index], bins=20, color="darkorange")
        axes[1, index].set_title(
            f"Столбец {index}: μ={stats.column_means[index]:.2f}, σ²={stats.column_variances[index]:.2f}"
        )
    figure.tight_layout()
    figure.savefig("histograms.png", dpi=120)
    print("Сохранено: histograms.png")


if __name__ == "__main__":
    _plot_histograms()
