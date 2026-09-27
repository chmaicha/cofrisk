import mlflow
import mlflow.xgboost

mlflow.set_experiment("CofRisk-Bankruptcy")

with mlflow.start_run():

    mlflow.log_params({
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "threshold": 0.20,
    })

    # entraînement de ton modèle
    # ...

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("pr_auc", pr_auc)
    mlflow.log_metric("mcc", mcc)

    mlflow.xgboost.log_model(
        model,
        "model",
    )