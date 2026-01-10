import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import norm, beta
from scipy.optimize import minimize_scalar


def normalize_probs(probs):
    """
    Normalize probabilities to [0, 1] using Min-Max scaling.
    Handles edge cases where all probabilities are identical.
    """
    min_p = np.min(probs)
    max_p = np.max(probs)
    if max_p == min_p:
        # If all probabilities are the same, return neutral value (0.5)
        return np.full_like(probs, 0.5)

    normalized = (probs - min_p) / (max_p - min_p)
    return np.clip(normalized, 0, 1)


def find_best_val_threshold(val_probs, val_labels, threshold_range=np.arange(0.01, 0.99, 0.001)):
    """
    Find the optimal threshold on the validation set that maximizes the F1 score.

    Args:
        val_probs: Validation predicted probabilities.
        val_labels: Validation ground truth labels.
        threshold_range: Range of thresholds to search.

    Returns:
        best_thresh: The optimal threshold.
        max_f1: The corresponding F1 score.
    """
    f1_list = []
    for thresh in threshold_range:
        preds = (val_probs > thresh).astype(int)
        f1 = f1_score(val_labels, preds)
        f1_list.append(f1)
    best_idx = np.argmax(f1_list)
    return threshold_range[best_idx], f1_list[best_idx]


def get_test_threshold(test_dist, val_q, dist_type):
    """
    Calculate the test set threshold based on the test distribution and the validation quantile.
    This implements the quantile matching calibration strategy.

    Args:
        test_dist: Fitted distribution object for the test set.
        val_q: The quantile associated with the optimal validation threshold.
        dist_type: Distribution type ("beta" or "norm").

    Returns:
        The calculated threshold alpha for the test set.
    """

    def objective(x):
        return (test_dist.cdf(x) - val_q) ** 2

    if dist_type == "beta":
        bounds = (1e-8, 1 - 1e-8)
    else:  # norm
        bounds = (test_dist.mean() - 3 * test_dist.std(),
                  test_dist.mean() + 3 * test_dist.std())

    res = minimize_scalar(objective, bounds=bounds, method='bounded')
    return np.clip(res.x, 1e-8, 1 - 1e-8)


def fit_distribution(probs, dist_type="beta"):
    """
    Fit a probability distribution (Beta or Normal) to the predicted probabilities.

    Args:
        probs: Array of probabilities (0-1).
        dist_type: "beta" (recommended) or "norm".

    Returns:
        dist: The frozen distribution object.
        params: The fitted parameters (e.g., alpha, beta or mu, sigma).
    """
    # Clip values to avoid fitting errors with Beta distribution
    probs_clipped = np.clip(probs, 1e-8, 1 - 1e-8)

    if dist_type == "beta":
        def neg_log_likelihood(params):
            a, b = params
            if a <= 0 or b <= 0:
                return 1e9
            return -np.sum(beta.logpdf(probs_clipped, a, b))

        # Initial parameter estimation
        mu = np.mean(probs_clipped)
        var = np.var(probs_clipped)
        a0 = mu * (mu * (1 - mu) / var - 1)
        b0 = (1 - mu) * (mu * (1 - mu) / var - 1)
        a0 = max(0.1, a0)
        b0 = max(0.1, b0)

        # Minimize negative log-likelihood
        from scipy.optimize import minimize
        res = minimize(neg_log_likelihood, [a0, b0], method='L-BFGS-B')
        a, b = res.x
        dist = beta(a, b)
        params = (a, b)

    elif dist_type == "norm":
        mu, sigma = norm.fit(probs_clipped)
        dist = norm(mu, sigma)
        params = (mu, sigma)

    else:
        raise ValueError("dist_type must be 'beta' or 'norm'")

    return dist, params