"""Calibration and selective-prediction metrics.

Every published number in the README comes from these functions, so each one
is small, pure, and unit-tested against a hand-computed value (see
tests/test_metrics.py). Nothing here imports torch: the metrics operate on
numpy arrays so they can be checked without a GPU.
"""
import numpy as np


def brier(p, y):
    """Mean squared error of the predicted probability."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def _confidence(p):
    return np.maximum(p, 1 - p)


def _correct(p, y):
    return ((np.asarray(p) >= 0.5).astype(int) == np.asarray(y)).astype(float)


def ece(p, y, bins=15):
    """Expected calibration error over `bins` equal-width confidence bins."""
    p, y = np.asarray(p, float), np.asarray(y)
    conf, correct = _confidence(p), _correct(p, y)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        hi = edges[i + 1] if i < bins - 1 else edges[i + 1] + 1e-12
        m = (conf >= edges[i]) & (conf < hi)
        if m.sum():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def auc(p, y):
    """Probability an unsafe row outranks a safe one. 0.5 = zero discrimination.

    This is the CEILING on what any threshold, prior correction or monotone
    rescaling can achieve. A model with AUC 0.58 cannot beat a 60-66% majority
    floor by any choice of cut point, so measuring AUC first tells you whether a
    threshold problem is even addressable before you spend a run tuning one.
    """
    p, y = np.asarray(p, float), np.asarray(y, float)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    gt = np.sum(pos[:, None] > neg[None, :])
    eq = np.sum(pos[:, None] == neg[None, :])
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def aurc(p, y):
    """Area under the risk-coverage curve. Lower is better; 0 is perfect."""
    p, y = np.asarray(p, float), np.asarray(y)
    wrong = 1.0 - _correct(p, y)
    order = np.argsort(-_confidence(p), kind="stable")
    risk = np.cumsum(wrong[order]) / np.arange(1, len(wrong) + 1)
    return float(np.mean(risk))


def coverage_at(p, y, budget=0.05):
    """Largest prefix (by descending confidence) whose error rate stays within
    `budget`. Returns (coverage_fraction, n_items).

    This is the operational number for a guard: it answers "how often may I
    trust the model at a k% error budget?" rather than "how often is it right?"
    """
    p, y = np.asarray(p, float), np.asarray(y)
    wrong = (1.0 - _correct(p, y)).astype(int)
    order = np.argsort(-_confidence(p), kind="stable")
    w = wrong[order]
    # The risk-coverage curve is NOT monotone: adding items can lower the
    # running mean. A budget is only honoured if EVERY prefix up to k is
    # within budget, so track the running maximum and stop at the first
    # violation. Returning the largest k whose own mean is within budget
    # would silently overstate coverage.
    running_max = 0.0
    best = 0
    for k in range(1, len(w) + 1):
        running_max = max(running_max, w[:k].mean())
        if running_max <= budget:
            best = k
        else:
            break
    return best / len(w), best


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


#: Public alias. eval_bench needs logit space to apply an intercept, and reaching
#: for the private `_logit` across module boundaries is what produced a NameError
#: in the first place -- an eval that died after a completed GPU-hour of training.
logit = _logit


def apply_temperature(p, t):
    """Rescale probabilities by a single temperature."""
    return 1.0 / (1.0 + np.exp(-_logit(p) / float(t)))


#: Temperature search bounds. The ceiling is deliberately far beyond any
#: plausible optimum: a fit that lands exactly ON the ceiling has not converged,
#: it has been truncated, and a truncated fit reports a calibration number that
#: looks measured. R-Judge fitted exactly 20.00 against the old ceiling of 20.
TEMP_MIN = 0.25
TEMP_MAX = 200.0


def fit_temperature_info(p, y, lo=TEMP_MIN, hi=TEMP_MAX, step=0.05):
    """Fit one scalar temperature by minimising log-loss; report the bound.

    Returns {"T", "nll", "at_bound"}. at_bound is True when the optimum is at
    the low or high end of the grid, which means the search was truncated and
    the returned T should not be read as a converged optimum.
    """
    p, y = np.asarray(p, float), np.asarray(y, float)
    lg = _logit(p)
    best_t, best_nll = 1.0, np.inf
    for t in np.arange(lo, hi + 1e-9, step):
        q = np.clip(1.0 / (1.0 + np.exp(-lg / t)), 1e-6, 1 - 1e-6)
        nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    at_bound = bool(best_t <= lo + step or best_t >= hi - step)
    # Distinguish the two ways to be at a bound. Hitting the ceiling means the
    # optimum is "as flat as possible": the model's probabilities carry LESS
    # information than the labels, so T -> infinity and the fitted value is a
    # truncation, not a measurement. Hitting the floor means they carry more.
    # Collapsing both into one boolean hid which one was happening.
    return {"T": best_t, "nll": float(best_nll), "at_bound": at_bound,
            "at_ceiling": bool(best_t >= hi - step),
            "at_floor": bool(best_t <= lo + step),
            "brier": float(np.mean((np.clip(p, 0, 1) - y) ** 2))}


def fit_temperature(p, y, lo=TEMP_MIN, hi=TEMP_MAX, step=0.05):
    """Fit one scalar temperature on held-out data by minimising log-loss."""
    return fit_temperature_info(p, y, lo, hi, step)["T"]


def gate(p, y, hi=0.7, lo=0.3):
    """Turn p(fail) into a three-way decision and score it.

    block  : p >= hi        refuse to run
    ask    : lo <= p < hi   surface to the human
    allow  : p <  lo        run unattended

    The thresholds live in code, not in the model: the model supplies a
    probability, the caller owns the policy.
    """
    p, y = np.asarray(p, float), np.asarray(y)
    block = p >= hi
    ask = (p >= lo) & (p < hi)
    allow = p < lo
    fails = int((y == 1).sum())
    caught = int((block & (y == 1)).sum())
    n_safe = int((y == 0).sum())
    return {
        "block_rate": float(block.mean()),
        "ask_rate": float(ask.mean()),
        "allow_rate": float(allow.mean()),
        "fail_recall": caught / fails if fails else 0.0,
        "block_precision": caught / int(block.sum()) if block.sum() else 0.0,
        "false_block_rate": int((block & (y == 0)).sum()) / n_safe if n_safe else 0.0,
    }


def summarize(p, y):
    """Every metric reported in the README, as one dict."""
    p, y = np.asarray(p, float), np.asarray(y)
    return {
        "n": int(len(y)),
        "acc": float((_correct(p, y)).mean()),
        "brier": brier(p, y),
        "ece": ece(p, y),
        "aurc": aurc(p, y),
        "auc": auc(p, y),
        "cov5": coverage_at(p, y, 0.05)[0],
        "cov10": coverage_at(p, y, 0.10)[0],
        "cov20": coverage_at(p, y, 0.20)[0],
    }


# ------------------------------------------------ divergence and memorisation

def _clip(p):
    return np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)


def kl_bernoulli(p, q):
    """KL(p || q) for Bernoulli distributions, in nats."""
    p, q = float(_clip(p)), float(_clip(q))
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def _cross_entropy(p, y):
    p, y = _clip(p), np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def kl_from_prior(p, y):
    """Information gained over the base-rate prior, in nats.

    Defined as  CE(prior) - CE(model):  how many nats the model saves compared
    with a constant predictor that always emits the base rate.

      > 0  better than the prior  (the model knows something)
      = 0  exactly the prior       (a constant predictor scores 0 whatever
                                    constant it picks -- confidence is not
                                    information)
      < 0  WORSE than the prior    (confidently wrong)

    The first implementation of this averaged a per-row log ratio, which is not
    a divergence and returned -0.51 for a constant predictor. A test asserting
    that a constant predictor scores 0.0 caught it.
    """
    p = _clip(p)
    y = np.asarray(y, float)
    prior = float(_clip(y.mean()))
    return _cross_entropy(np.full(len(y), prior), y) - _cross_entropy(p, y)


def memorisation_gap(p_train, y_train, p_test, y_test, kl_threshold=0.25):
    """Compare how much the model learned on train versus held-out data.

    A falling training loss is not evidence of learning: a model that memorises
    drives it to ~0 while generalising no better than chance. The signal is the
    GAP. Train divergence much larger than held-out divergence means the model
    fits rows it has seen and carries little information about rows it has not.

    `flagged` is True when train is markedly better than held-out, which is the
    memorisation signature rather than a merely-difficult task.
    """
    kl_tr = kl_from_prior(p_train, y_train)
    kl_te = kl_from_prior(p_test, y_test)

    def acc(p, y):
        p, y = _clip(p), np.asarray(y, float)
        return float(np.mean((p >= 0.5) == (y >= 0.5)))

    a_tr, a_te = acc(p_train, y_train), acc(p_test, y_test)
    gap = kl_tr - kl_te
    return {
        "kl_train": kl_tr,
        "kl_test": kl_te,
        "kl_gap": gap,
        "acc_train": a_tr,
        "acc_test": a_te,
        "acc_gap": a_tr - a_te,
        "flagged": bool(gap > kl_threshold),
    }
