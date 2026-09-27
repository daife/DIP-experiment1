"""Image-conditioned HOG kernel initialization with local Ridge refinement."""
from pathlib import Path
import cv2
import joblib
import numpy as np
from .landmark_regression import features, normalized_image


def global_features(images):
    result = []
    cell_ids = (np.arange(64)[:, None]//8)*8 + np.arange(64)[None, :]//8
    for image in images:
        small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        gx, gy = np.zeros_like(small), np.zeros_like(small)
        gx[:, 1:-1] = small[:, 2:]-small[:, :-2]
        gy[1:-1] = small[2:]-small[:-2]
        bins = np.floor(np.mod(np.arctan2(gy, gx), np.pi)*9/np.pi).astype(int).clip(0, 8)
        cells = np.bincount((cell_ids*9+bins).ravel(), weights=np.hypot(gx, gy).ravel(),
                            minlength=64*9).reshape(8, 8, 9)
        blocks = np.asarray([cells[y:y+2, x:x+2].ravel() for y in range(7) for x in range(7)])
        blocks /= np.sqrt((blocks*blocks).sum(1, keepdims=True)+1e-6)
        result.append(blocks.ravel())
    return np.asarray(result, dtype=np.float64)


class HogLandmarkRegressor:
    def __init__(self, initializer, offsets, coefficients):
        self.initializer = initializer
        self.offsets = offsets
        self.coefficients = coefficients

    def predict_normalized(self, images):
        shape = self.initializer.predict(global_features(images)).reshape(-1, 28, 2)
        for coef in self.coefficients:
            x = features(images, shape, self.offsets)
            shape += (np.column_stack([np.ones(len(x)), x])@coef).reshape(-1, 28, 2)
            shape = np.clip(shape, -.25, 1.25)
        return shape

    def predict_image(self, bgr, bbox):
        box = np.asarray(bbox, dtype=np.float64)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        shape = self.predict_normalized(normalized_image(gray, box)[None])[0]
        return shape*(box[2:]-box[:2])+box[:2]

    def save(self, path: Path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path):
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ValueError('Wrong landmark model type')
        return model
