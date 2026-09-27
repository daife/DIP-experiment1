"""Check a 511-bin split search prototype against exhaustive weighted error.

This benchmark does not change the running trainer or deployed inference.
"""
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.weak_tree_training import _best_split


def histogram_split(values, labels, weights, mask):
    selected = np.flatnonzero(mask)
    if not len(selected):
        return 255, 0.
    bins = values[selected].astype(np.int32)+255
    positive = np.bincount(bins, weights=weights[selected]*labels[selected], minlength=511)
    negative = np.bincount(bins, weights=weights[selected]*(1-labels[selected]), minlength=511)
    occupied = np.flatnonzero(np.bincount(bins,minlength=511))
    cp, cn = np.cumsum(positive), np.cumsum(negative)
    errors = np.minimum(cp,cn)+np.minimum(cp[-1]-cp,cn[-1]-cn)
    index = occupied[np.argmin(errors[occupied])]
    return int(index-255),float(errors[index])


def direct_error(values, labels, weights, mask, threshold):
    left = mask & (values<=threshold)
    right = mask & ~left
    return sum(min(weights[branch & (labels==1)].sum(),weights[branch & (labels==0)].sum()) for branch in (left,right))


def main():
    rng = np.random.default_rng(20260926)
    threshold_differences = 0
    for trial in range(120):
        n = int(rng.integers(1,200))
        values = rng.integers(-255,256,n,dtype=np.int16)
        labels = rng.integers(0,2,n,dtype=np.int8)
        weights = rng.random(n)
        weights /= weights.sum()
        mask = rng.random(n)>.3
        if trial % 10 == 0:
            weights[:] = 1/n
        if trial % 12 == 0:
            labels[:] = trial%2
        if trial == 0:
            mask[:] = False
        threshold,error = histogram_split(values,labels,weights,mask)
        old_threshold,old_error = _best_split(values,labels,weights,mask)
        candidates = np.unique(values[mask])
        expected = min((direct_error(values,labels,weights,mask,t) for t in candidates),default=0.)
        assert abs(error-expected)<1e-12
        assert abs(direct_error(values,labels,weights,mask,threshold)-expected)<1e-12
        assert abs(old_error-expected)<1e-12
        if threshold != old_threshold:
            threshold_differences += 1
    n = 30000
    values = rng.integers(-255,256,n,dtype=np.int16)
    labels = rng.integers(0,2,n,dtype=np.int8)
    weights = rng.random(n)
    weights /= weights.sum()
    mask = rng.random(n)>.3
    timings={}
    for name,method in [('sort',_best_split),('histogram',histogram_split)]:
        start=perf_counter()
        for _ in range(200):
            method(values,labels,weights,mask)
        timings[name]=perf_counter()-start
    report={'seed':20260926,'randomized_brute_force_checks':120,'threshold_differences':threshold_differences,
            'max_allowed_weighted_error_difference':1e-12,'benchmark_samples':n,'repeats':200,'seconds':timings,
            'speed_ratio':timings['sort']/timings['histogram'],
            'scope':'split-search only; concurrent training load; floating reduction/ties may alter chosen thresholds; not model equivalence or deployment speed',
            'running_trainer_changed':False}
    (ROOT/'results/cascade_split_histogram_benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
