import numpy as np

from src.landmark_regression import features, fit_stage, make_offsets


def test_masked_ridge_ignores_unreliable_target():
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    residual = np.zeros((4, 56))
    residual[:, 0] = [2.0, 4.0, 6.0, 10000.0]
    mask = np.ones((4, 28), dtype=bool)
    mask[3, 0] = False
    coef = fit_stage(x, residual, mask, alpha=0.01)
    predicted = np.column_stack([np.ones(4), x]) @ coef
    assert np.max(np.abs(predicted[:3, 0] - [2, 4, 6])) < 0.02


def test_local_pixel_difference_is_shape_dependent():
    image = np.tile(np.arange(32, dtype=np.float32), (32, 1))[None] / 31
    offsets = np.zeros((28, 1, 2, 2))
    offsets[:, :, 1, 0] = 0.1
    shape = np.full((1, 28, 2), 0.5)
    first = features(image, shape, offsets)
    shape[:, :, 0] = 0.95
    second = features(image, shape, offsets)
    assert first.shape == (1, 28)
    assert not np.allclose(first, second)  # right sample clips at the image edge
    assert np.array_equal(make_offsets(42), make_offsets(42))
