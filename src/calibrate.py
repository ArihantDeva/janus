"""Post-hoc temperature scaling, fitted on data the model never trained on.

WHY THIS IS NOT OPTIONAL
The done-criterion is ECE <= 0.05. A distilled classifier is systematically
overconfident -- it was trained by KL-matching a teacher whose median confidence is
0.93, so it learns to be peaked, and a peaked model on a 3-way problem is wrong more
often than its peak suggests. Measured on the smoke run: ECE 0.1375 with a reliability
curve reading `conf 0.387 / acc 0.587` in one bin and `conf 0.409 / acc 0.427` in the
next -- non-monotone, i.e. confidence carries almost no information as a ranking at
that granularity. Thresholding that (which is exactly what the Muse hook does) would
fire on rows the model is merely opinionated about.

WHAT TEMPERATURE SCALING FIXES AND WHAT IT DOES NOT
It rescales logits by a single scalar T and cannot change the argmax, so accuracy is
untouched by construction. It fixes calibration only. That is precisely the claim being
made (ECE), and claiming a calibration win from a method that cannot alter accuracy is
honest; claiming an accuracy win from it would be a lie.

WHY THE FITTING SET MUST BE DISJOINT FROM BOTH TRAIN AND EVAL
T on the eval set is fitted-on-test: it would report an ECE the model does not have on
new data. Fitting on train measures the model's overconfidence on rows it memorised,
which is not the deployed condition. So the corpus is split three ways by group --
train / calib / eval -- and `assert_disjoint` refuses otherwise. This is the same
group-disjoint discipline the train/eval split already uses, applied to the third split
that calibration quietly introduces.

WHY NELDER-MEAD ON ECE AND NOT ON NLL
Optimising NLL for T is the textbook choice and it usually helps ECE, but the criterion
is ECE, so this fits ECE directly and reports the NLL cost alongside. When two
objectives disagree the reported one wins; the loser is printed, not hidden.
"""

from __future__ import annotations

import numpy as np


def apply_temperature(logits, temperature):
    """logits / T, then softmax. T > 1 softens, T < 1 sharpens."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def expected_calibration_error(P, y, bins=10):
    """Standard ECE: |confidence - accuracy| averaged over confidence bins.

    Bins are equal-width on confidence, weighted by bin population, which is the
    definition the criterion uses. Empty bins contribute nothing.
    """
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y)
    if y.dtype.kind not in "iu":
        raise TypeError(
            f"labels must be integer class indices, got {y.dtype}; string labels make "
            f"`pred == y` always False and report 0% accuracy")
    conf = P.max(1)
    correct = (P.argmax(1) == y).astype(np.float64)
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf >= lo) & (conf < hi) if b < bins - 1 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def nll(P, y):
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y)
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean())


def _nelder_mead(f, x0, step=0.5, iters=400, tol=1e-10):
    """Minimal Nelder-Mead in one dimension. One scalar parameter does not justify a
    dependency, and scipy is not in this project's dependency set.

    Written out explicitly rather than golfed: the first draft of the contraction step
    computed the reflected point twice and used it as both the candidate and the
    fallback, which silently makes the contraction a no-op. The coarser grid search in
    `fit_temperature` would have masked it.
    """
    pts = np.array([float(x0), float(x0) + step])
    vals = np.array([f(pts[0]), f(pts[1])])
    for _ in range(iters):
        if vals[0] > vals[1]:
            pts, vals = pts[::-1], vals[::-1]
        if abs(vals[1] - vals[0]) < tol:
            break
        centroid = pts[0]
        xr = centroid + (centroid - pts[1])          # reflection
        fr = f(xr)
        if fr < vals[0]:                             # better than best: expand
            xe = centroid + 2.0 * (centroid - pts[1])
            fe = f(xe)
            pts, vals = (np.array([xe, pts[0]]), np.array([fe, vals[0]])) \
                if fe < fr else (np.array([xr, pts[0]]), np.array([fr, vals[0]]))
        elif fr < vals[1]:                           # accept reflection
            pts, vals = np.array([xr, pts[0]]), np.array([fr, vals[0]])
        else:                                        # contract
            xc = centroid + 0.5 * (pts[1] - centroid)
            fc = f(xc)
            if fc < vals[1]:
                pts, vals = np.array([xc, pts[0]]), np.array([fc, vals[0]])
            else:                                    # shrink toward best
                xs = centroid + 0.5 * (pts[1] - centroid)
                pts, vals = np.array([centroid, xs]), np.array([vals[0], f(xs)])
    return float(pts[0]), float(vals[0])


def fit_temperature(logits, y, objective="ece", lo=0.05, hi=10.0):
    """Find the T minimising ECE (default) or NLL on the given rows."""
    if objective not in ("ece", "nll"):
        raise ValueError(f"objective must be 'ece' or 'nll', got {objective}")
    metric = expected_calibration_error if objective == "ece" else nll

    def f(t):
        t = float(np.clip(t, lo, hi))
        return metric(apply_temperature(logits, t), y)

    best_t, best_v = _nelder_mead(f, 1.0)
    # a coarse grid guards against Nelder-Mead stalling on a flat ECE surface, which
    # is common: ECE is piecewise constant because it depends only on bin membership
    for cand in np.exp(np.linspace(np.log(lo), np.log(hi), 60)):
        v = f(cand)
        if v < best_v:
            best_t, best_v = float(cand), v
    return float(np.clip(best_t, lo, hi))


def assert_disjoint(*splits):
    """Refuse to calibrate on rows the model trained on, or on the eval rows.

    A calibration set that is not disjoint silently produces the best possible ECE
    number and the worst possible deployed model.
    """
    groups = [set(s.get("group") for s in split) for split in splits]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            overlap = groups[i] & groups[j]
            if overlap:
                raise AssertionError(
                    f"splits {i} and {j} share {len(overlap)} groups "
                    f"(e.g. {sorted(map(str, overlap))[:3]}); calibration must be "
                    f"disjoint from both train and eval")


def select_threshold(P, y, target_accuracy=None, min_coverage=0.5):
    """Pick the operating threshold on the CALIBRATION split.

    The hook thresholds on confidence, so the threshold is a real hyperparameter and
    picking it from the eval set is the same defect as fitting temperature there: the
    reported operating point would not be the deployed one.

    Two ways to choose, in order:
      target_accuracy   the lowest threshold whose answered rows reach that accuracy.
                        This is the product question -- "how much can I answer at 85%".
      min_coverage      with no accuracy target, the threshold maximising accuracy
                        subject to answering at least this fraction.

    Returns (threshold, coverage, accuracy) measured on the rows passed in, so the
    caller can print what it fitted rather than trusting it.
    """
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y)
    conf = P.max(1)
    correct = (P.argmax(1) == y)
    best = (1.01, 0.0, 0.0)
    for tau in np.arange(0.30, 0.96, 0.01):
        m = conf >= tau
        cov = float(m.mean())
        if cov < min_coverage or not m.any():
            continue
        acc = float(correct[m].mean())
        if target_accuracy is not None:
            if acc >= target_accuracy:
                return float(tau), cov, acc
        elif acc > best[2] or (acc == best[2] and cov > best[1]):
            best = (float(tau), cov, acc)
    if target_accuracy is not None:
        return None, 0.0, 0.0        # target unreachable at the coverage floor
    return best
