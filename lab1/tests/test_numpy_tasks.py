"""Тесты второй части лабораторной №1 (NumPy)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grader_contracts.numpy_tasks import (
    BinarizeInput,
    ChessInput,
    EllipseInput,
    MatrixInput,
    MatrixVectorBatchInput,
    OneHotInput,
    RandomMatrixInput,
    RectangleInput,
    TimeSeriesInput,
)
from numpy_tasks import (
    analyze_time_series,
    binarize,
    chess,
    draw_ellipse,
    draw_rectangle,
    matrix_statistics,
    one_hot,
    sum_prod,
    unique_columns,
    unique_rows,
)


def test_sum_prod():
    matrices = [np.eye(2), np.eye(2)]
    vectors = [np.array([1, 2]), np.array([3, 4])]
    result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    assert result.shape == (2, 1)
    assert np.array_equal(result.ravel(), [4, 6])


def test_binarize_does_not_mutate():
    source = np.array([[0.2, 0.6], [0.5, 0.9]])
    result = binarize(BinarizeInput(source, 0.5))
    assert result.tolist() == [[0, 1], [0, 1]]
    assert source.tolist() == [[0.2, 0.6], [0.5, 0.9]]


def test_unique_rows_and_columns():
    matrix = np.array([[1, 1, 2], [3, 3, 3]])
    assert unique_rows(MatrixInput(matrix)) == [[1, 2], [3]]
    assert unique_columns(MatrixInput(np.array([[1, 2], [1, 5]]))) == [[1], [2, 5]]


def test_matrix_statistics_shapes_and_reproducibility():
    stats_a = matrix_statistics(RandomMatrixInput(3, 4, seed=7))
    stats_b = matrix_statistics(RandomMatrixInput(3, 4, seed=7))
    assert stats_a.matrix.shape == (3, 4)
    assert stats_a.row_means.shape == (3,)
    assert stats_a.column_variances.shape == (4,)
    assert np.allclose(stats_a.matrix, stats_b.matrix)


def test_chess():
    expected = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    assert chess(ChessInput(3, 4, 0, 1)).tolist() == expected


def test_draw_shapes_shape_and_colors():
    rect = draw_rectangle(RectangleInput(4, 2, 10, 10, (255, 0, 0), (0, 0, 0)))
    assert rect.shape == (10, 10, 3)
    assert tuple(rect[5, 5]) == (255, 0, 0)
    ellipse = draw_ellipse(EllipseInput(3, 2, 11, 11, (255, 0, 0), (0, 0, 0)))
    assert ellipse.shape == (11, 11, 3)
    assert tuple(ellipse[5, 5]) == (255, 0, 0)


def test_analyze_time_series():
    stats = analyze_time_series(TimeSeriesInput([1, 3, 2, 5, 4], 2))
    assert list(stats.local_maxima_indices) == [1, 3]
    assert list(stats.local_minima_indices) == [2]
    assert np.allclose(stats.moving_average, [2.0, 2.5, 3.5, 4.5])
    assert stats.mean == 3.0


def test_one_hot():
    assert one_hot(OneHotInput([0, 2, 3, 0])).tolist() == [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    assert one_hot(OneHotInput([0, 1], class_count=4)).shape == (2, 4)
