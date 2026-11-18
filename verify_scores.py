#!/usr/bin/env python3
"""Reproduce submitted scores from anonymized saved complete-trajectory measurements."""
from pathlib import Path
import json
import numpy as np
from arcon.quality import scores
ROOT=Path(__file__).resolve().parent

def main():
    meta=json.loads((ROOT/'examples/frozen_scores.json').read_text())
    maximum=0.;count=0
    with np.load(ROOT/'examples/frozen_measurements.npz',allow_pickle=False) as arrays:
        for row in meta['cases']:
            result=scores(arrays[row['key']])
            delta=max(abs(result[k]-row[k]) for k in ['JAR','JCV'])
            maximum=max(maximum,delta);count+=1
        for row in meta['pairs']:
            result=scores(np.stack([arrays[k] for k in row['members']]),arrays[row['rho_key']])
            delta=max(abs(result[k]-row[k]) for k in ['JAR-4R','JCV-4R'])
            maximum=max(maximum,delta);count+=1
    if maximum>1e-8:raise ValueError(f'Submitted scores differ by {maximum} points')
    print(json.dumps({'comparisons':count,'maximum_score_difference_points':maximum,'source':'complete saved paper trajectories'}))

if __name__=='__main__':main()
