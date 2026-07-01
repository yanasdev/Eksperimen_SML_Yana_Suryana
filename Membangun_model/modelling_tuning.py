import json
import os
from pathlib import Path

import dagshub
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, train_test_split


BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = "House_Price_Prediction_Tuning"
MLFLOW_URI = "http://127.0.0.1:5000"
DAGSHUB_OWNER = "yanas.dev"
DAGSHUB_REPO = "Eksperimen_MSML_Yana_Suryana"


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def get_dagshub_token() -> str:
    token = os.getenv("DAGSHUB_USER_TOKEN") or os.getenv("DAGSHUB_TOKEN")
    if token:
        return token

    token = input("Masukkan DagsHub Token Anda untuk menyimpan artefak online: ").strip()
    if not token:
        raise ValueError("Token DagsHub tidak boleh kosong.")
    return token


def configure_tracking() -> str:
    local_uri = MLFLOW_URI
    remote_uri = f"https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}.mlflow"

    load_env_file(BASE_DIR / ".env")
    token = get_dagshub_token()
    os.environ["DAGSHUB_USER_TOKEN"] = token
    os.environ["DAGSHUB_TOKEN"] = token

    try:
        dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True, dvc=False, patch_mlflow=True)
        mlflow.set_tracking_uri(remote_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        print(f"MLflow Tracking diset ke DagsHub: {remote_uri}")
        return remote_uri
    except Exception as exc:
        print(f"Tidak bisa terhubung ke DagsHub: {exc}")
        print(f"Menggunakan MLflow lokal: {local_uri}")
        mlflow.set_tracking_uri(local_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        return local_uri


def save_feature_importance(model: RandomForestRegressor, feature_names: list[str], output_base: Path) -> tuple[Path, Path]:
    importance_df = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    importance_df = importance_df.sort_values("importance", ascending=False)

    csv_path = output_base.with_suffix(".csv")
    png_path = output_base.with_suffix(".png")
    importance_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"].head(15).tolist(), importance_df["importance"].head(15).tolist())
    plt.title("Top 15 Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    return csv_path, png_path


def save_predictions(predictions: pd.Series, y_test: pd.Series, output_path: Path) -> None:
    evaluation_df = pd.DataFrame({"actual": y_test, "predicted": predictions})
    evaluation_df.to_csv(output_path, index=False)


def save_metadata(metadata: dict, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


tracking_uri = configure_tracking()

load_env_file(BASE_DIR / ".env")

df = pd.read_csv(BASE_DIR / "clean_data.csv")
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 5, 10],
}

artifact_folder = BASE_DIR / "artifacts"
artifact_folder.mkdir(exist_ok=True)

for params in ParameterGrid(param_grid):
    run_name = f"rf_{params['n_estimators']}_{params['max_depth']}"
    with mlflow.start_run(run_name=run_name) as run:
        model = RandomForestRegressor(random_state=42, **params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        mlflow.log_params(params)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        config_path = artifact_folder / f"config_{run.info.run_id}.json"
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(params, file, indent=2)

        prediction_path = artifact_folder / f"predictions_{run.info.run_id}.csv"
        save_predictions(pd.Series(predictions), y_test, prediction_path)

        feature_path_base = artifact_folder / f"feature_importance_{run.info.run_id}"
        feature_csv_path, feature_png_path = save_feature_importance(model, X.columns.tolist(), feature_path_base)

        metadata_path = artifact_folder / f"run_metadata_{run.info.run_id}.json"
        save_metadata(
            {
                "tracking_uri": tracking_uri,
                "experiment_name": EXPERIMENT_NAME,
                "run_name": run_name,
                "params": params,
                "metrics": {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2},
                "artifact_count": 5,
            },
            metadata_path,
        )

        mlflow.log_artifact(str(config_path))
        mlflow.log_artifact(str(prediction_path))
        mlflow.log_artifact(str(feature_csv_path))
        mlflow.log_artifact(str(feature_png_path))
        mlflow.log_artifact(str(metadata_path))
        mlflow.sklearn.log_model(model, "model")

        print(f"Run {run.info.run_id} selesai dengan R2={r2:.4f}")
