# Multifactor Linear Regression Analysis

A Python implementation of a two-factor linear econometric model using the Ordinary Least Squares (OLS) method. The project estimates regression coefficients, evaluates the relative impact of each factor, and assesses model quality through the variance-covariance matrix of estimates.

> **Course project:** *Digital Information-Analytical Systems*
> **Author:** Vanin Dmytro
> **University:** V. N. Karazin Kharkiv National University

---

## Problem statement

Given an empirical dataset with two explanatory variables (X₁, X₂) and one response variable (Y), build a linear regression model:

$$Y = b_0 + b_1 \cdot X_1 + b_2 \cdot X_2$$

Estimate the coefficients, analyze the influence of each factor, and verify the quality of the fit.

## Method

Coefficients are computed by the closed-form OLS solution:

$$\mathbf{B} = (X^T X)^{-1} X^T Y$$

where `X` is the design matrix with a leading column of ones (for the intercept), and `Y` is the response vector.

The script then computes:

| Metric | Formula | Meaning |
|---|---|---|
| Standardized coefficients β | $\beta_i = b_i \cdot (s_{x_i} / s_y)$ | Relative impact in standard deviations |
| Elasticity coefficients E | $E_i = b_i \cdot (\bar{x_i} / \bar{y})$ | Percentage change in Y per 1% change in Xᵢ |
| Residual variance σ² | $\hat\sigma^2 = \mathrm{SSR} / (n - m)$ | Spread of residuals around the fit |
| Covariance matrix K | $K = \hat\sigma^2 \cdot (X^T X)^{-1}$ | Precision of coefficient estimates |

## Project structure

```
econometrics-analytics/
├── nain.py            Main analysis script
├── data.json          Input dataset (X1, X2, Y)
├── requirements.txt   Python dependencies
└── README.md          This file
```

## Requirements

- Python 3.10 or newer
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python regression.py
```

The script reads `data.json` from the project root and prints the full report to stdout. To analyze a different dataset, simply replace the contents of `data.json` keeping the same structure:

```json
{
  "description": "Your dataset description",
  "X1": [...],
  "X2": [...],
  "Y":  [...]
}
```

## Sample output

```
============================================================
MULTIFACTOR LINEAR REGRESSION — RESULTS
============================================================

1. REGRESSION EQUATION
------------------------------------------------------------
   Y = 5.1736 + 1.3722 * X1 + 0.8523 * X2

2. FACTOR INFLUENCE ANALYSIS
------------------------------------------------------------
   Standardized coefficient beta_1 = 0.5961
   Standardized coefficient beta_2 = 0.4165
   Elasticity coefficient    E_1   = 0.4380
   Elasticity coefficient    E_2   = 0.4502

3. VARIANCE-COVARIANCE MATRIX
------------------------------------------------------------
   Estimated variance of residuals (sigma^2) = 0.0197
   Covariance matrix K:
      0.1249  0.0088  -0.0104
      0.0088  0.0015  -0.0013
      -0.0104 -0.0013  0.0012
```

## Findings

For the dataset in `data.json`, the fitted model is:

$$Y = 5.1736 + 1.3722 \cdot X_1 + 0.8523 \cdot X_2$$

**Interpretation of coefficients:**

- The intercept **b₀ = 5.1736** is the expected value of Y when both factors equal zero.
- Both slopes are positive, indicating that X₁ and X₂ have a direct positive effect on Y.
- A unit increase in X₁ (with X₂ fixed) raises Y by **1.3722** on average.
- A unit increase in X₂ (with X₁ fixed) raises Y by **0.8523** on average.

**Relative impact of factors:**

- Standardized coefficients (β₁ = 0.5961, β₂ = 0.4165) show that X₁ has a stronger absolute influence on the variation of Y than X₂.
- Elasticity coefficients (E₁ = 0.4380, E₂ = 0.4502) are very close, meaning the *relative* (percentage-based) impact of both factors is essentially the same — a 1% increase in either factor raises Y by roughly 0.44–0.45%.

**Model quality:**

- The residual variance σ² = 0.0197 is small, indicating a tight fit of the model to the data.
- The diagonal elements of the covariance matrix K (0.1249, 0.0015, 0.0012) are small, confirming high precision of the coefficient estimates.

**Conclusion:** the constructed two-factor linear econometric model is adequate and suitable for analyzing the dependence of Y on X₁ and X₂, as well as for forecasting Y at given factor levels.

## License

MIT
