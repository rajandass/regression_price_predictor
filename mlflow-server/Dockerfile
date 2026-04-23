FROM python:3.10-slim

WORKDIR /app

RUN pip install mlflow azure-storage-blob psycopg2-binary

EXPOSE 5000

CMD ["sh", "-c", "mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination $ARTIFACT_ROOT \
  --serve-artifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --workers 1 \
  --allowed-hosts '*' \
  --cors-allowed-origins '*' \
  --gunicorn-opts '--timeout 120'"]