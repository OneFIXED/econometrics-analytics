"""
Multifactor Linear Regression Analysis
=======================================

Implementation of a two-factor linear econometric model using the
Ordinary Least Squares (OLS) method.

Model: Y = b0 + b1*X1 + b2*X2

Computes:
- Regression coefficients (b0, b1, b2)
- Standardized coefficients (beta)
- Elasticity coefficients (E)
- Variance of residuals (sigma^2)
- Variance-covariance matrix of estimates

Author: Vanin Dmytro
Course: Digital Information-Analytical Systems
University: V. N. Karazin Kharkiv National University
"""

import numpy as np
import json
from pathlib import Path


def load_data(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["X1"]), np.array(data["X2"]), np.array(data["Y"])


def build_design_matrix(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Build the design matrix X with a column of ones for the intercept.
    """
    n = len(x1)
    return np.column_stack((np.ones(n), x1, x2))


def estimate_coefficients(x_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Estimate regression coefficients using OLS:
        B = (X^T * X)^(-1) * X^T * Y
    """
    x_t = x_matrix.T
    x_t_x_inv = np.linalg.inv(x_t @ x_matrix)
    return x_t_x_inv @ x_t @ y


def analyze_factor_influence(
    b1: float, b2: float,
    x1: np.ndarray, x2: np.ndarray, y: np.ndarray
) -> dict:
    """
    Compute standardized coefficients (beta) and elasticity coefficients.
    """
    mean_y, mean_x1, mean_x2 = y.mean(), x1.mean(), x2.mean()
    std_y, std_x1, std_x2 = y.std(), x1.std(), x2.std()

    return {
        "beta_1": b1 * (std_x1 / std_y),
        "beta_2": b2 * (std_x2 / std_y),
        "elasticity_1": b1 * (mean_x1 / mean_y),
        "elasticity_2": b2 * (mean_x2 / mean_y),
    }


def variance_covariance_matrix(
    b: np.ndarray, x_matrix: np.ndarray, y: np.ndarray
) -> tuple[float, np.ndarray]:
    """
    Estimate variance of residuals and the variance-covariance matrix
    of coefficient estimates.

    sigma^2 = SSR / (n - m), where m = number of parameters
    K = sigma^2 * (X^T * X)^(-1)
    """
    n = len(y)
    m = x_matrix.shape[1]

    y_hat = x_matrix @ b
    residuals = y - y_hat
    ssr = np.sum(residuals ** 2)

    sigma2 = ssr / (n - m)
    x_t_x_inv = np.linalg.inv(x_matrix.T @ x_matrix)
    k_matrix = sigma2 * x_t_x_inv

    return sigma2, k_matrix


def print_report(b: np.ndarray, influence: dict, sigma2: float, k_matrix: np.ndarray):
    b0, b1, b2 = b

    print("=" * 60)
    print("MULTIFACTOR LINEAR REGRESSION — RESULTS")
    print("=" * 60)

    print("\n1. REGRESSION EQUATION")
    print("-" * 60)
    print(f"   Y = {b0:.4f} + {b1:.4f} * X1 + {b2:.4f} * X2")

    print("\n2. FACTOR INFLUENCE ANALYSIS")
    print("-" * 60)
    print(f"   Standardized coefficient beta_1 = {influence['beta_1']:.4f}")
    print(f"   Standardized coefficient beta_2 = {influence['beta_2']:.4f}")
    print(f"   Elasticity coefficient    E_1   = {influence['elasticity_1']:.4f}")
    print(f"   Elasticity coefficient    E_2   = {influence['elasticity_2']:.4f}")

    print("\n3. VARIANCE-COVARIANCE MATRIX")
    print("-" * 60)
    print(f"   Estimated variance of residuals (sigma^2) = {sigma2:.4f}")
    print("   Covariance matrix K:")
    for row in k_matrix:
        formatted = "\t".join(f"{val:.4f}" for val in row)
        print(f"      {formatted}")
    print()


def main():
    data_path = Path(__file__).parent / "data.json"
    x1, x2, y = load_data(str(data_path))

    x_matrix = build_design_matrix(x1, x2)
    b = estimate_coefficients(x_matrix, y)
    influence = analyze_factor_influence(b[1], b[2], x1, x2, y)
    sigma2, k_matrix = variance_covariance_matrix(b, x_matrix, y)

    print_report(b, influence, sigma2, k_matrix)


if __name__ == "__main__":
    main()