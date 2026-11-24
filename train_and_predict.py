import platform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
np.random.seed(SEED)

# Solo variables disponibles al momento de publicar un auto
# (excluye 'sold_price_usd' como objetivo y 'price_drop_pct' por disponibilidad posterior)
FEATURE_COLS = [
    "brand",
    "year",
    "fuel",
    "transmission",
    "engine_cc",
    "city",
    "color",
    "mileage_km",
]
NUM_COLS = ["year", "engine_cc", "mileage_km"]
CAT_COLS = ["brand", "fuel", "transmission", "city", "color"]
TARGET_COL = "sold_price_usd"
MIN_VALID_PRICE_USD = 300.0


def clean_listings(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
  data = df.drop_duplicates(subset=["listing_id"]).copy()

  text_cols = ["brand", "fuel", "transmission", "city", "color", "mileage_unit"]
  for col in text_cols:
    if col in data.columns:
      data[col] = data[col].astype(str).str.strip()
      data.loc[data[col].isin(["nan", "None", ""]), col] = np.nan

  data["fuel"] = data["fuel"].str.lower().replace({"nafta": "gasoline"})
  data["transmission"] = data["transmission"].str.lower()
  data["mileage_unit"] = data["mileage_unit"].str.lower()

  data["year"] = pd.to_numeric(data["year"], errors="coerce")
  data["engine_cc"] = pd.to_numeric(data["engine_cc"], errors="coerce")
  data["mileage"] = pd.to_numeric(data["mileage"], errors="coerce")

  is_miles = data["mileage_unit"] == "miles"
  data["mileage_km"] = np.where(
      is_miles, data["mileage"] * 1.60934, data["mileage"]
  )

  global_trans = data["transmission"].mode(dropna=True).iloc[0]
  data["transmission"] = (
      data["transmission"]
      .fillna(
          data.groupby(["brand", "fuel"])["transmission"].transform(
              lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
          )
      )
      .fillna(global_trans)
  )

  global_cc = data["engine_cc"].median()
  data["engine_cc"] = (
      data["engine_cc"]
      .fillna(data.groupby(["brand", "fuel"])["engine_cc"].transform("median"))
      .fillna(global_cc)
  )

  if is_train:
    data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
    data = data.dropna(subset=[TARGET_COL] + FEATURE_COLS).copy()
    data = data[data[TARGET_COL] >= MIN_VALID_PRICE_USD].copy()
    return data[["listing_id"] + FEATURE_COLS + [TARGET_COL]].reset_index(
        drop=True
    )

  data = data.dropna(subset=FEATURE_COLS).copy()
  return data[["listing_id"] + FEATURE_COLS].reset_index(drop=True)


def main() -> None:
  print(
      f"Python: {platform.python_version()} | pandas: {pd.__version__} | "
      f"numpy: {np.__version__} | scikit-learn: {sklearn.__version__} | "
      f"matplotlib: {matplotlib.__version__}"
  )

  train_raw = pd.read_csv("listings_train.csv")
  new_raw = pd.read_csv("listings_new.csv")

  train_df = clean_listings(train_raw, is_train=True)
  new_df = clean_listings(new_raw, is_train=False)

  X_full = train_df[FEATURE_COLS]
  y_full = train_df[TARGET_COL]
  y_log_full = np.log1p(y_full)

  # Reservar 20% para una única evaluación final sin tocarlo antes
  X_train_cv, X_test, y_train_log, y_test_log = train_test_split(
      X_full, y_log_full, test_size=0.20, random_state=SEED
  )
  y_test_actual = np.expm1(y_test_log)

  preprocessor = ColumnTransformer(
      transformers=[
          (
              "num",
              Pipeline([
                  ("imputer", SimpleImputer(strategy="median")),
                  ("scaler", StandardScaler()),
              ]),
              NUM_COLS,
          ),
          (
              "cat",
              Pipeline([
                  ("imputer", SimpleImputer(strategy="most_frequent")),
                  ("onehot", OneHotEncoder(handle_unknown="ignore")),
              ]),
              CAT_COLS,
          ),
      ]
  )

  base_models = {
      "Ridge": Ridge(alpha=1.0, random_state=SEED),
      "GradientBoosting": GradientBoostingRegressor(random_state=SEED),
      "RandomForest": RandomForestRegressor(
          n_estimators=150, random_state=SEED, n_jobs=-1
      ),
  }

  kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
  cv_rows = []
  for name, estimator in base_models.items():
    mae_folds, mape_folds = [], []
    for tr_idx, va_idx in kf.split(X_train_cv):
      pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
      pipe.fit(X_train_cv.iloc[tr_idx], y_train_log.iloc[tr_idx])
      preds = np.expm1(pipe.predict(X_train_cv.iloc[va_idx]))
      truth = np.expm1(y_train_log.iloc[va_idx])
      mae_folds.append(mean_absolute_error(truth, preds))
      mape_folds.append(
          mean_absolute_percentage_error(truth, preds) * 100.0
      )
    cv_rows.append({
        "Model": name,
        "MAE_mean": round(float(np.mean(mae_folds)), 2),
        "MAE_std": round(float(np.std(mae_folds)), 2),
        "MAPE_mean": round(float(np.mean(mape_folds)), 2),
        "MAPE_std": round(float(np.std(mape_folds)), 2),
    })

  cv_df = pd.DataFrame(cv_rows).sort_values("MAE_mean").reset_index(drop=True)
  print("\n5-Fold CV Results (80% Train, log-price target):")
  print(cv_df.to_string(index=False))

  # Ajuste de alpha sobre Ridge mediante 5-Fold CV
  alpha_grid = [0.1, 1.0, 5.0, 10.0, 25.0]
  tune_rows = []
  for alpha_val in alpha_grid:
    mae_f, mape_f = [], []
    for tr_idx, va_idx in kf.split(X_train_cv):
      ridge_est = Ridge(alpha=alpha_val, random_state=SEED)
      pipe = Pipeline([("prep", preprocessor), ("model", ridge_est)])
      pipe.fit(X_train_cv.iloc[tr_idx], y_train_log.iloc[tr_idx])
      preds = np.expm1(pipe.predict(X_train_cv.iloc[va_idx]))
      truth = np.expm1(y_train_log.iloc[va_idx])
      mae_f.append(mean_absolute_error(truth, preds))
      mape_f.append(mean_absolute_percentage_error(truth, preds) * 100.0)
    tune_rows.append({
        "alpha": alpha_val,
        "CV_MAE": round(float(np.mean(mae_f)), 2),
        "CV_MAPE": round(float(np.mean(mape_f)), 2),
    })

  tune_df = (
      pd.DataFrame(tune_rows).sort_values("CV_MAE").reset_index(drop=True)
  )
  print("\nRidge Alpha Tuning (5-Fold CV):")
  print(tune_df.to_string(index=False))

  best_alpha = float(tune_df.loc[0, "alpha"])

  # Evaluación única en el 20% reservado
  best_pipe = Pipeline([
      ("prep", preprocessor),
      ("model", Ridge(alpha=best_alpha, random_state=SEED)),
  ])
  best_pipe.fit(X_train_cv, y_train_log)
  test_preds = np.expm1(best_pipe.predict(X_test))
  test_mae = mean_absolute_error(y_test_actual, test_preds)
  test_mape = mean_absolute_percentage_error(y_test_actual, test_preds) * 100.0
  test_r2 = r2_score(y_test_actual, test_preds)

  print(
      f"\nHold-Out Test (20%) | Best Model Ridge (alpha={best_alpha}) -> "
      f"MAE: {test_mae:.2f} USD | MAPE: {test_mape:.2f}% | R2: {test_r2:.4f}"
  )

  # Reentrenamiento con todo el set limpio y predicción sobre listings_new.csv
  final_pipe = Pipeline([
      ("prep", preprocessor),
      ("model", Ridge(alpha=best_alpha, random_state=SEED)),
  ])
  final_pipe.fit(X_full, y_log_full)
  new_preds = np.expm1(final_pipe.predict(new_df[FEATURE_COLS]))

  predictions_out = pd.DataFrame({
      "listing_id": new_df["listing_id"],
      "predicted_price": np.round(new_preds, 2),
  })
  predictions_out.to_csv("predictions.csv", index=False)
  print("Saved predictions.csv with shape:", predictions_out.shape)

  from sklearn.inspection import permutation_importance

  perm = permutation_importance(
      best_pipe,
      X_test,
      y_test_log,
      n_repeats=15,
      random_state=SEED,
      n_jobs=-1,
  )
  imp_vals = np.maximum(perm.importances_mean, 0.0)
  total_imp = float(imp_vals.sum())
  imp_shares = (
      (imp_vals / total_imp) * 100.0 if total_imp > 0 else np.zeros_like(imp_vals)
  )

  imp_series = pd.Series(imp_shares, index=FEATURE_COLS).sort_values(
      ascending=True
  )
  plt.figure(figsize=(8, 4.5))
  imp_series.plot(kind="barh", color="#1f77b4")
  plt.title("Permutation Importance in Used-Car Price Prediction (Ridge)")
  plt.xlabel("Relative Importance Share (%)")
  plt.ylabel("Feature")
  plt.grid(True, axis="x", linestyle="--", alpha=0.4)
  plt.tight_layout()
  plt.savefig("feature_importance.png", dpi=150)
  plt.close()

  # Gráfico 2: Predicho vs. Real en Test (20%)
  plt.figure(figsize=(7, 5.5))
  plt.scatter(
      y_test_actual,
      test_preds,
      alpha=0.45,
      s=18,
      color="#1f77b4",
      label="Test listings (20%)",
  )
  max_val = max(float(y_test_actual.max()), float(test_preds.max()))
  plt.plot(
      [0, max_val],
      [0, max_val],
      linestyle="--",
      color="#d62728",
      linewidth=1.5,
      label="Ideal 1:1 fit",
  )
  plt.title("Predicted vs. Actual Price (Hold-Out Test Set - Ridge)")
  plt.xlabel("Actual Sold Price (USD)")
  plt.ylabel("Predicted Price (USD)")
  plt.legend()
  plt.grid(True, linestyle="--", alpha=0.4)
  plt.tight_layout()
  plt.savefig("predicted_vs_actual.png", dpi=150)
  plt.close()
  print("Saved charts: feature_importance.png, predicted_vs_actual.png")


if __name__ == "__main__":
  main()