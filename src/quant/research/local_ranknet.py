"""Research RankNet pairs, including overlapping score neighbourhoods.

Scores used to construct training pairs must be chronological out-of-fold
predictions. Pair labels are outcomes; neither outcomes nor pair membership are
features at inference. Hessians are a positive diagonal upper bound on the
pairwise logistic Hessian, suitable for tree boosting.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.special import expit


def percentile_scores(dates, scores) -> np.ndarray:
    frame = pd.DataFrame({'date': np.asarray(dates), 'score': np.asarray(scores)})
    return frame.groupby('date').score.rank(method='average', pct=True).to_numpy()


@dataclass(frozen=True)
class Pairs:
    left: np.ndarray
    right: np.ndarray
    target: np.ndarray
    weight: np.ndarray


def make_pairs(dates, scores, outcome, bandwidth=None, per_row=8, seed=73) -> Pairs:
    """Sample unordered pairs within date and exact percentile distance.

    Neighbourhoods overlap: no arbitrary bin edge or transitive chaining. All
    retained pairs satisfy the distance bound; a day has equal total loss weight.
    """
    dates, scores, outcome = map(np.asarray, (dates, scores, outcome))
    if not (len(dates) == len(scores) == len(outcome)):
        raise ValueError('Mismatched pair input lengths')
    if not np.isfinite(scores).all() or not np.isfinite(outcome).all():
        raise ValueError('Pairs require finite OOF scores and mature outcomes')
    if bandwidth is not None and not 0 < bandwidth <= 1:
        raise ValueError('Bandwidth must be in (0, 1]')
    rng = np.random.default_rng(seed)
    left, right, weights = [], [], []
    for ids in pd.Series(np.arange(len(dates))).groupby(dates, sort=True):
        ix = ids[1].to_numpy()
        ix = ix[np.argsort(scores[ix], kind='stable')]
        if len(ix) < 2:
            continue
        # Vectorized random proposals from the exact forward neighbourhood.
        upper = (np.searchsorted(scores[ix], scores[ix] + bandwidth, side='right')
                 if bandwidth is not None else np.full(len(ix), len(ix)))
        lower = np.arange(len(ix)) + 1
        width = upper - lower
        anchors = np.repeat(np.flatnonzero(width > 0), per_row)
        dest = lower[anchors] + (rng.random(len(anchors)) * width[anchors]).astype(int)
        edges = np.unique(np.column_stack([ix[anchors], ix[dest]]), axis=0)
        edges = edges[outcome[edges[:, 0]] != outcome[edges[:, 1]]]
        if not len(edges):
            continue
        left.append(edges[:, 0]); right.append(edges[:, 1])
        weights.append(np.full(len(edges), 1 / len(edges)))
    if not left:
        raise ValueError('No informative pairs')
    a, b, w = np.concatenate(left), np.concatenate(right), np.concatenate(weights)
    w *= len(dates) / w.sum()
    return Pairs(a, b, (outcome[a] > outcome[b]).astype(float), w)


def derivatives(prediction: np.ndarray, pairs: Pairs):
    probability = expit(prediction[pairs.left] - prediction[pairs.right])
    gradient = pairs.weight * (probability - pairs.target)
    curvature = pairs.weight * probability * (1 - probability)
    n = len(prediction)
    g = (np.bincount(pairs.left, weights=gradient, minlength=n)
         - np.bincount(pairs.right, weights=gradient, minlength=n))
    h = 2 * (np.bincount(pairs.left, weights=curvature, minlength=n)
             + np.bincount(pairs.right, weights=curvature, minlength=n))
    return g, np.maximum(h, 1e-8)


def cross_date_pairs(dates, scores, outcome, bandwidth, per_row=8, seed=73) -> Pairs:
    """Exact raw-score neighbourhood across DIFFERENT dates, equal anchor-date weight.

    Absolute net outcomes of a single fixed holding horizon are compared. All
    outcomes must have matured by this model's fit boundary. Scores are OOF net
    return predictions, so bandwidth is expressed in return units (not ranks).
    """
    dates, scores, outcome = map(np.asarray, (dates, scores, outcome))
    if not (len(dates)==len(scores)==len(outcome)) or bandwidth <= 0:
        raise ValueError('Invalid cross-date inputs')
    if not np.isfinite(scores).all() or not np.isfinite(outcome).all():
        raise ValueError('Finite OOF scores and mature outcomes required')
    order=np.argsort(scores,kind='stable'); sorted_scores=scores[order]
    low=np.searchsorted(sorted_scores,sorted_scores-bandwidth,side='left')
    high=np.searchsorted(sorted_scores,sorted_scores+bandwidth,side='right')
    rng=np.random.default_rng(seed)
    anchors=np.repeat(np.arange(len(order)),per_row)
    other=low[anchors]+(rng.random(len(anchors))*(high-low)[anchors]).astype(int)
    a,b=order[anchors],order[other]
    keep=(dates[a]!=dates[b])&(outcome[a]!=outcome[b])
    edges=np.unique(np.column_stack([a[keep],b[keep]]),axis=0)
    if not len(edges): raise ValueError('No informative cross-date pairs')
    a,b=edges[:,0],edges[:,1]
    _, inverse, counts=np.unique(dates[a],return_inverse=True,return_counts=True)
    weight=1/counts[inverse].astype(float);weight*=len(scores)/weight.sum()
    return Pairs(a,b,(outcome[a]>outcome[b]).astype(float),weight)


def mature_before(frame: pd.DataFrame, boundary) -> pd.Series:
    boundary = pd.Timestamp(boundary)
    return frame.date.lt(boundary) & frame.maturity.lt(boundary) & frame.outcome.notna()


def choose_blend(validation: dict[str, pd.DataFrame], tolerance=0.) -> str:
    """Maximize conservative net return without sacrificing validation wins."""
    base = validation['base50']
    admitted = [name for name, day in validation.items()
                if day.stress_win.mean() >= base.stress_win.mean() - tolerance
                and day.stress.mean() >= base.stress.mean()]
    return max(admitted, key=lambda name: validation[name].stress.mean())
