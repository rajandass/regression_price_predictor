from mlflow.tracking import MlflowClient

client = MlflowClient()

experiments = client.search_experiments()

for exp in experiments:
    print(exp.experiment_id, exp.name)