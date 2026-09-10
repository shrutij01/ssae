# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Disentanglement, Completeness and Informativeness.

Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).
"""

import numpy as np
import scipy.stats
from joblib import Parallel, delayed
from sklearn import ensemble


def compute_importance_gbt(
    x_train,
    y_train,
    x_test,
    y_test,
    discrete_factors=None,
    random_state=42,
    n_jobs=1,
):
    """Compute importance based on gradient boosted trees.

    Supports both discrete and continuous factors:
    - Discrete: Uses GradientBoostingClassifier (informativeness = accuracy)
    - Continuous: Uses GradientBoostingRegressor (informativeness = R²)

    Args:
        x_train: Training codes of shape (num_codes, n_train).
        y_train: Training factors of shape (num_factors, n_train).
        x_test: Test codes of shape (num_codes, n_test).
        y_test: Test factors of shape (num_factors, n_test).
        discrete_factors: Optional list of bools indicating which factors are
            discrete. If None, auto-detects: factors with integer-like values
            are discrete, otherwise continuous.
        random_state: Seed used by each gradient-boosted tree.
        n_jobs: Number of factor probes to fit concurrently. Each factor is an
            independent supervised problem, so parallel execution does not
            change the fitted estimators or their deterministic ordering.

    Returns:
        importance_matrix: Array of shape (num_codes, num_factors).
        train_score: Mean informativeness on training set.
        test_score: Mean informativeness on test set.
    """
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1 or a positive integer")

    def fit_factor(i):
        if discrete_factors is not None:
            is_discrete = discrete_factors[i]
        else:
            factor_values = y_train[i, :]
            is_discrete = np.allclose(factor_values, np.round(factor_values))

        if is_discrete:
            model = ensemble.GradientBoostingClassifier(
                random_state=random_state
            )
            model.fit(x_train.T, y_train[i, :])
            importance = np.abs(model.feature_importances_)
            train_score = np.mean(model.predict(x_train.T) == y_train[i, :])
            test_score = np.mean(model.predict(x_test.T) == y_test[i, :])
        else:
            model = ensemble.GradientBoostingRegressor(
                random_state=random_state
            )
            model.fit(x_train.T, y_train[i, :])
            importance = np.abs(model.feature_importances_)
            train_score = model.score(x_train.T, y_train[i, :])
            test_score = model.score(x_test.T, y_test[i, :])

        return i, importance, float(train_score), float(test_score)

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(fit_factor)(i) for i in range(num_factors)
    )
    importance_matrix = np.zeros(
        shape=[num_codes, num_factors], dtype=np.float64
    )
    train_scores = []
    test_scores = []
    for i, importance, train_score, test_score in results:
        importance_matrix[:, i] = importance
        train_scores.append(train_score)
        test_scores.append(test_score)

    return importance_matrix, np.mean(train_scores), np.mean(test_scores)


def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    num_factors = importance_matrix.shape[1]
    if num_factors <= 1:
        # With 1 factor, disentanglement is trivially 1.0
        return np.ones(importance_matrix.shape[0])
    return 1.0 - scipy.stats.entropy(
        importance_matrix.T + 1e-11, base=num_factors
    )


def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.0:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()
    return np.sum(per_code * code_importance)


def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    num_codes = importance_matrix.shape[0]
    if num_codes <= 1:
        return np.ones(importance_matrix.shape[1])
    return 1.0 - scipy.stats.entropy(
        importance_matrix + 1e-11, base=num_codes
    )


def completeness(importance_matrix):
    """Compute completeness of the representation."""
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.0:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor * factor_importance)


def compute_dci(
    codes: np.ndarray,
    factors: np.ndarray,
    train_fraction: float = 0.8,
    random_state: int | None = 42,
    discrete_factors: list | None = None,
) -> dict[str, float]:
    """Compute DCI metrics from codes and ground truth factors.

    Args:
        codes: SAE activations of shape (n_samples, n_codes).
        factors: Ground truth factors of shape (n_samples, n_factors).
            For binary labels, reshape to (n_samples, 1).
        train_fraction: Fraction of data used for training GBT.
        random_state: Random seed for train/test split.
        discrete_factors: Optional list of bools per factor.

    Returns:
        Dict with keys: disentanglement, completeness,
        informativeness_train, informativeness_test.
    """
    n = codes.shape[0]
    if factors.ndim == 1:
        factors = factors[:, np.newaxis]

    n_train = int(n * train_fraction)

    if random_state is not None:
        rng = np.random.RandomState(random_state)
        indices = rng.permutation(n)
    else:
        indices = np.arange(n)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    # Transpose to (features, samples) for DisentanglementLib convention
    codes_train = codes[train_idx].T
    codes_test = codes[test_idx].T
    factors_train = factors[train_idx].T
    factors_test = factors[test_idx].T

    importance_matrix, train_acc, test_acc = compute_importance_gbt(
        codes_train, factors_train,
        codes_test, factors_test,
        discrete_factors=discrete_factors,
        random_state=random_state,
    )

    return {
        "disentanglement": float(disentanglement(importance_matrix)),
        "completeness": float(completeness(importance_matrix)),
        "informativeness_train": float(np.clip(train_acc, 0.0, 1.0)),
        "informativeness_test": float(np.clip(test_acc, 0.0, 1.0)),
    }
