import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# MLflow Setup
# -----------------------------
mlflow.set_experiment("house_price_prediction")

# -----------------------------
# Load Clean Data
# -----------------------------
df = pd.read_csv("house-price-dataset-of-india/clean_data.csv")

# Normalize column names
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# Drop weak feature (if exists)
if "condition_of_the_house" in df.columns:
    df = df.drop(["condition_of_the_house"], axis=1)

# Feature engineering safely
if "built_year" in df.columns:
    df["house_age"] = 2025 - df["built_year"]
    df = df.drop("built_year", axis=1)

# Optional feature
if "living_area" in df.columns and "number_of_bedrooms" in df.columns:
    df["area_per_bedroom"] = df["living_area"] / df["number_of_bedrooms"]

print("Data loaded:", df.shape)

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop("price", axis=1)

# Log transform target
y = np.log1p(df["price"])

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# -----------------------------
# Start MLflow Run
# -----------------------------
with mlflow.start_run():

    # Model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
# -----------------------------
# Model Pipeline
# -----------------------------
pipeline = Pipeline([
    ("model", model)
])

# -----------------------------
# Train Model
# -----------------------------
pipeline.fit(X_train, y_train)

# -----------------------------
# Predict (Test Set)
# -----------------------------
y_pred_log = pipeline.predict(X_test)

# Convert back to original scale
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

# -----------------------------
# Evaluation
# -----------------------------
mae = mean_absolute_error(y_test_actual, y_pred)
mse = mean_squared_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)

print("\n📊 Model Performance (Test Set):")
print(f"MAE: {mae:,.2f}")
print(f"MSE: {mse:,.2f}")
print(f"R2 Score: {r2:.4f}")

# -----------------------------
# Cross Validation
# -----------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")

mae_scores = cross_val_score(
    pipeline, X, y,
    cv=cv,
    scoring="neg_mean_absolute_error"
)
cv_r2 = r2_scores.mean()
print("\n📊 Cross Validation Results:")
print("R2 Scores:", r2_scores)
print("Mean CV R2:", cv_r2)
print("Std Dev:", r2_scores.std())

print("Mean CV MAE (log scale):", -mae_scores.mean())

# -----------------------------
# Log to MLflow
# -----------------------------
mlflow.log_param("n_estimators", 200)
mlflow.log_param("max_depth", 10)

mlflow.log_metric("mae", mae)
mlflow.log_metric("mse", mse)
mlflow.log_metric("r2", r2)
mlflow.log_metric("cv_r2", cv_r2)

# Log model
mlflow.sklearn.log_model(pipeline, "model")

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(pipeline, "model.pkl")

print("\n✅ Model logged to MLflow and saved")

run_id = mlflow.active_run().info.run_id
model_uri = f"runs:/{run_id}/model"

mlflow.register_model(model_uri, "house-price-model")