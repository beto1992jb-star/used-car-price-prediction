# Used-Car Price Prediction Pipeline

End-to-end reproducible machine learning pipeline to estimate used-car prices (`sold_price_usd`) from listing attributes available at publication time, predicting on `log1p(sold_price_usd)` and transforming back with `expm1`.

## How to Run

1. Place `listings_train.csv` and `listings_new.csv` in the working directory.
2. Run the script:
   ```bash
   python train_and_predict.py
   ```
3. Outputs generated in the working directory:
   - `predictions.csv` (`listing_id`, `predicted_price` for new listings)
   - `feature_importance.png`
   - `predicted_vs_actual.png`

## Libraries & Versions
Printed directly at runtime by `train_and_predict.py`:
- `Python: 3.13.18 | pandas: 2.2.3 | numpy: 2.4.6 | scikit-learn: 1.6.1 | matplotlib: 3.10.0`

Note: numbers may vary slightly between library versions.

## Data Cleaning & Leakage Prevention
- **Excluded post-publication / target fields:** `sold_price_usd` is the training target and `price_drop_pct` is excluded because it is not available when a new listing is created.
- **Typo filtering on historical prices:** Excluded 30 implausible low-price records below `300 USD` (ranging from `11 USD` to `180 USD`, before prices jump to `439+ USD`) that distorted percentage errors.
- **Distance standardization:** Converted `miles` to kilometers (`* 1.60934`) into `mileage_km`.
- **Category normalization:** Standardized `transmission` case (`manual`, `automatic`) and mapped `nafta` to `gasoline` in `fuel`.
- **Missing values:** Imputed `transmission` (mode by `brand` and `fuel`, with global fallback) and `engine_cc` (median by `brand` and `fuel`, with global fallback).
- **Reproducibility:** Fixed `SEED = 42` across splits, cross-validation, models, and tuning.

## Model Comparison (5-Fold CV on 80% Training Split, Log-Price Target)

| Model | MAE Mean ± Std (USD) | MAPE Mean ± Std (%) |
| :--- | :---: | :---: |
| Ridge (`alpha=1.0`) | 556.14 ± 14.47 | 8.90% ± 0.14% |
| GradientBoosting (`default`) | 632.08 ± 40.91 | 9.99% ± 0.21% |
| RandomForest (`n_estimators=150`) | 718.96 ± 45.74 | 11.41% ± 0.16% |

### Selected Final Model
`Ridge` is selected as the final model because it achieves the lowest MAE (`556.14 ± 14.47 USD`) and lowest MAPE (`8.90% ± 0.14%`) in 5-fold cross-validation. Percentage depreciation across vehicle age and mileage aligns closely with a linear pattern on the log-price scale (`log1p`), so a regularized linear model generalizes cleanly without overfitting and is straightforward to explain. Tuning `alpha` via 5-fold CV showed that `alpha` had no practical effect on cross-validation error; single evaluation on the untouched 20% hold-out test set achieved **MAE = 547.79 USD**, **MAPE = 8.68%**, and **R² = 0.9675** (numbers may vary slightly between library versions).

## Limitations
- High-value listings have fewer historical examples, leading to wider dispersion at upper price levels.
- Static listing fields do not capture real-time market shifts or physical vehicle condition.