# Ames Housing — Regression with Regularization

Predicting house sale prices on the Ames Housing dataset using Linear Regression, Ridge, and Lasso. This project covers the full supervised learning pipeline: data cleaning, categorical encoding, feature selection via multicollinearity analysis, scaling, regularization, and hyperparameter tuning with GridSearchCV.

---

## Overview

The Ames Housing dataset contains 80 features describing residential properties in Ames, Iowa. The goal is to predict `SalePrice` from a combination of numerical and categorical features. This project compares a manually engineered 9-feature linear model against regularized models trained on all 294 encoded features, demonstrating why regularization outperforms manual feature selection at scale.

---

## Dataset

**Ames Housing Dataset** — residential property sales in Ames, Iowa

| Property | Detail |
|---|---|
| Source | `mod2_ames.csv` |
| Observations | 2,930 |
| Original features | 80 (37 numerical, 43 categorical) |
| Target | `SalePrice` (continuous) |
| After encoding | 294 features |

---

## Data Preparation

**Missing value handling:**
- 5 columns with >50% missing data dropped (Pool QC, Misc Feature, Alley, Fence, Mas Vnr Type)
- Garage and basement features filled with `'None'` — missing indicates absence of feature
- Fireplace quality filled with `'None'` — missing indicates no fireplace
- Electrical filled with mode (1 missing value)

**Encoding:**
- All 38 remaining categorical columns one-hot encoded using `OneHotEncoder`
- Final feature matrix: 294 columns, all numeric

**Note on encoding order:** Encoding was applied before the train/test split in this assignment. In production, the encoder should be fit only on training data and applied to test data to prevent data leakage — ensuring the model learns nothing from test set category distributions.

**Train/test split:** 80% train (2,344 samples) / 20% test (586 samples)

---

## Feature Selection (Linear Regression)

For the baseline linear model, 9 features were selected from training data correlations, resolving 6 high-correlation feature pairs:

| Pair Removed | Correlation | Feature Kept |
|---|---|---|
| Garage Cars vs Garage Area | 0.884 | Garage Cars |
| Gr Liv Area vs TotRms AbvGrd | 0.806 | Gr Liv Area |
| Total Bsmt SF vs 1st Flr SF | 0.800 | Total Bsmt SF |
| Year Built vs Foundation_PConc | 0.651 | Year Built |
| Gr Liv Area vs Full Bath | 0.623 | Gr Liv Area |
| Year Built vs Year Remod/Add | 0.594 | Year Built |

**Final 9 features:** Overall Qual, Gr Liv Area, Garage Cars, Total Bsmt SF, Year Built, Mas Vnr Area, Fireplaces, BsmtFin Type 1_GLQ, Exter Qual_Gd

Maximum pairwise correlation in final set: **0.592** (below 0.60 threshold)

---

## Models

### Linear Regression (9 selected features)
- Manual feature selection via correlation and multicollinearity analysis
- No regularization, no scaling required

### Ridge Regression (all 294 features)
- L2 regularization — shrinks all coefficients toward zero
- Retains all features with reduced influence
- Tested alpha range: 0.001 to 1,000 (100 values, log scale)

### Lasso Regression (all 294 features)
- L1 regularization — performs automatic feature selection
- Drives irrelevant coefficients to exactly zero
- Tested alpha range: 0.001 to 1,000 (100 values, log scale)

All features scaled with `StandardScaler` before Ridge/Lasso training (required for fair regularization across features with different scales).

---

## Results

| Model | Alpha | Train R² | Test R² |
|---|---|---|---|
| Linear Regression (9 features) | N/A | 0.7957 | 0.8166 |
| Ridge — manual search | 123.28 | 0.9268 | 0.8959 |
| **Lasso — manual search** | **247.71** | **0.9260** | **0.8987 ✓** |
| Ridge — GridSearchCV | 572.24 | — | 0.8921 |
| Lasso — GridSearchCV | 572.24 | — | 0.8969 |

**Best model: Lasso with α=247.71**, achieving **Test R² = 0.8987** (89.9% of variance explained).

Lasso automatically eliminated 123 of 294 features, using 171 to make predictions. The 8.2 percentage point improvement over the manually selected linear model demonstrates the value of regularization when working with high-dimensional encoded feature spaces.

---

## Key Findings

**Manual search vs GridSearchCV:** Both approaches identified similar optimal alpha values (~250–570), but GridSearchCV's cross-validated selection is more robust in practice — it averages performance across 5 folds rather than optimizing for a single train/test split. In this case, manual search found a marginally better alpha for this specific split, but GridSearchCV is the preferred approach for production use.

**Regularization vs manual feature selection:** Ridge and Lasso consistently outperformed the 9-feature linear model by ~8 percentage points, capturing information from encoded categorical features (neighborhood, quality ratings, foundation type) that would otherwise be discarded.

---

## Tech Stack

- Python, scikit-learn, pandas, NumPy, Matplotlib
- `LinearRegression`, `Ridge`, `Lasso`, `GridSearchCV`
- `OneHotEncoder`, `StandardScaler`
- `train_test_split`, `r2_score`

---

## Repository Structure

```
ames-housing-regression/
├── notebook.ipynb    # Full pipeline: cleaning, encoding, modeling, evaluation
├── README.md
```

---

## Dataset Source

Dean De Cock (2011). Ames, Iowa: Alternative to the Boston Housing Data Set. *Journal of Statistics Education*, 19(3).
