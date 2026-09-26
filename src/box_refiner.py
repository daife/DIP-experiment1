"""Optional train-fitted HOG box regression for Cascade proposals."""
from pathlib import Path
import joblib
import numpy as np
from .candidate_verifier import features


def transform_boxes(boxes, predictions, width, height):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != boxes.shape or not np.isfinite(predictions).all():
        raise ValueError('box deltas must be finite (N,4) values')
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    sizes = boxes[:, 2:] - boxes[:, :2]
    delta = np.clip(predictions, [-1.5,-1.5,-1,-1], [1.5,1.5,1,1])
    center = centers + delta[:, :2]*sizes
    size = sizes*np.exp(delta[:, 2:])
    adjusted = np.rint(np.concatenate((center-size/2,center+size/2),axis=1)).astype(np.int32)
    adjusted[:,[0,2]] = np.clip(adjusted[:,[0,2]],0,width)
    adjusted[:,[1,3]] = np.clip(adjusted[:,[1,3]],0,height)
    return adjusted, np.all(adjusted[:,2:]>adjusted[:,:2],axis=1)


class BoxRefiner:
    def __init__(self, path: str | Path):
        self.model = joblib.load(path)
        if getattr(self.model,'n_features_in_',None) != 324 or np.shape(getattr(self.model,'coef_',None)) != (4,324):
            raise ValueError('invalid HOG box regression model')

    def refine(self, gray, boxes):
        if not len(boxes):
            return np.empty((0,4),dtype=np.int32), np.empty(0,dtype=bool)
        return transform_boxes(boxes,self.model.predict(features(gray,boxes)),gray.shape[1],gray.shape[0])
