"""Cascade training with equal initial mass for negative source categories.

Keeps Real AdaBoost updates and weak-tree search unchanged. The temporary
initial-weight override is local to this synchronous training process.
"""
import numpy as np
from src import cascade as core


def grouped_weights(labels,groups):
    labels=np.asarray(labels);groups=np.asarray(groups)
    if labels.shape!=groups.shape or labels.ndim!=1 or not np.all(np.isin(labels,(0,1))):
        raise ValueError('grouped labels must be aligned binary vectors')
    positive=labels==1;negative=~positive
    if not positive.any() or not negative.any() or not np.issubdtype(groups.dtype,np.integer) or np.any(groups[negative]<0):
        raise ValueError('need both classes and nonnegative integer negative groups')
    categories=np.unique(groups[negative])
    weights=np.zeros(len(labels),dtype=float)
    weights[positive]=.5/positive.sum()
    for category in categories:
        mask=negative&(groups==category)
        weights[mask]=.5/len(categories)/mask.sum()
    return weights


def fit_group_balanced_cascade(train_windows,train_labels,validation_windows,validation_labels,*,negative_groups,
                               seed,num_stages,num_trees,root_candidates,child_candidates,target_recall):
    labels=np.asarray(train_labels);groups=np.asarray(negative_groups);vy=np.asarray(validation_labels)
    grouped_weights(labels,groups)
    if len(train_windows)!=len(labels) or len(validation_windows)!=len(vy):
        raise ValueError('window and label lengths differ')
    train_alive=np.ones(len(labels),dtype=bool);validation_alive=np.ones(len(vy),dtype=bool)
    stages=[]
    for index in range(num_stages):
        selected=(labels==1)|((labels==0)&train_alive)
        cal=(vy==1)&validation_alive
        if not np.any(cal):
            raise ValueError('no calibration faces survive')
        selected_labels=labels[selected];selected_groups=groups[selected]
        original=core._balanced_weights
        def initialize(stage_labels):
            if not np.array_equal(stage_labels,selected_labels):
                raise ValueError('stage labels differ from grouped inputs')
            return grouped_weights(stage_labels,selected_groups)
        try:
            core._balanced_weights=initialize
            stage=core.fit_stage(train_windows[selected],selected_labels,validation_windows[cal],
                                  seed=seed+index*10000,num_trees=num_trees,root_candidates=root_candidates,
                                  child_candidates=child_candidates,target_recall=target_recall)
        finally:
            core._balanced_weights=original
        stages.append(stage)
        ids=np.flatnonzero(train_alive)
        train_alive[ids]=stage.scores(train_windows[ids])>=stage.threshold
        ids=np.flatnonzero(validation_alive)
        validation_alive[ids]=stage.scores(validation_windows[ids])>=stage.threshold
    return core.Cascade(tuple(stages))
