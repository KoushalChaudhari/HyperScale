import numpy as np

from evaluation.metrics import image_metrics


def test_identical_and_degraded_images_have_expected_metrics():
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    identical = image_metrics(image, image)
    assert abs(identical["ssim"] - 1.0) < 1e-6
    assert np.isinf(identical["psnr"])
    degraded = image.copy()
    degraded[4:20, 4:20] = 0
    scores = image_metrics(image, degraded)
    assert 0 <= scores["ssim"] < 1
    assert scores["psnr"] < identical["psnr"]
