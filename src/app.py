"""
Credit Score Prediction API
===========================

Este módulo implementa uma função AWS Lambda exposta via API Gateway.
Ele carrega um modelo treinado de score de crédito, recebe um payload
JSON, faz a predição e:

* Publica métricas customizadas no CloudWatch;
* Armazena o payload + predição em um arquivo CSV no S3 (um arquivo por dia).

O objetivo é rodar sem modificações tanto localmente quanto na AWS.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List

import boto3
import joblib

# ---------------------------------------------------------------------------
# Configuração de logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constantes e objetos globais (carregados apenas no *cold start*)
# ---------------------------------------------------------------------------
BUCKET_NAME = "fiap-10dtsr-mlops-trabalho-final"
S3_PREFIX = "credit-score-real-data"

MODEL_NAMESPACE = "CreditScoreModel"
FEATURES_NAMESPACE = "CreditScoreFeatures"

MODEL_PATH = "model/model.pkl"
MODEL_METADATA_PATH = "model/model_metadata.json"

try:
    MODEL = joblib.load(MODEL_PATH)
except FileNotFoundError:  # pragma: no cover
    logger.error("Model file %s not found.", MODEL_PATH)
    raise

with open(MODEL_METADATA_PATH, encoding="utf-8") as f:
    MODEL_INFO: Dict[str, Any] = json.load(f)

s3_client = boto3.client("s3")
cloudwatch_client = boto3.client("cloudwatch")

# ---------------------------------------------------------------------------
# Ordem das features (treinada pelo modelo)
# ---------------------------------------------------------------------------
FEATURE_ORDER: List[str] = [
    "Age",
    "Annual_Income",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Num_of_Delayed_Payment",
    "Credit_Utilization_Ratio",
    "Payment_of_Min_Amount",
    "Total_EMI_per_month",
    "Credit_History_Age_Formated",
    "Auto_Loan",
    "Credit-Builder_Loan",
    "Personal_Loan",
    "Home_Equity_Loan",
    "Mortgage_Loan",
    "Student_Loan",
    "Debt_Consolidation_Loan",
    "Payday_Loan",
    "Missed_Payment_Day",
]

# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _normalise_value(value: Any) -> float | int:
    """Converte valores str -> int/float conforme necessário."""
    if isinstance(value, str):
        return float(value) if "." in value else int(value)
    return value  # type: ignore[return-value]


def prepare_payload(data: Dict[str, Any]) -> List[float]:
    """Converte o dicionário de entrada na lista na ordem esperada pelo modelo."""
    processed: List[float] = []

    for key in FEATURE_ORDER:
        if key not in data:
            raise ValueError(f"Feature '{key}' missing from payload.")
        processed.append(_normalise_value(data[key]))

    return processed


def write_real_data_to_s3(data: Dict[str, Any], prediction: int) -> None:
    """
    Persiste o payload enriquecido em CSV no S3.

    Um arquivo por dia: YYYY-MM-DD_credit_score_prediction_data.csv
    """

    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y %H:%M")
    file_name = f"{now.strftime('%Y-%m-%d')}_credit_score_prediction_data.csv"
    key = f"{S3_PREFIX}/{file_name}"

    enriched = {
        **data,
        "credit_score_prediction": prediction,
        "timestamp": timestamp,
        "model_version": MODEL_INFO["version"],
    }

    # Tentamos baixar o arquivo existente para anexar a nova linha
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        existing_lines = response["Body"].read().decode("utf-8").splitlines()
        if not existing_lines:
            raise ValueError("Empty file retrieved from S3")
        header, *rows = existing_lines
    except s3_client.exceptions.NoSuchKey:  # type: ignore[attr-defined]
        header, rows = ",".join(enriched.keys()), []

    rows.append(",".join(map(str, enriched.values())))
    content = "\n".join([header, *rows])

    s3_client.put_object(Body=content, Bucket=BUCKET_NAME, Key=key)


def publish_metrics(data: Dict[str, Any], prediction: float) -> None:
    """Publica métricas customizadas no CloudWatch."""
    cloudwatch_client.put_metric_data(
        MetricData=[
            {
                "MetricName": "CreditScorePrediction",
                "Value": prediction,
                "Unit": "None",
            }
        ],
        Namespace=MODEL_NAMESPACE,
    )

    for key, value in data.items():
        cloudwatch_client.put_metric_data(
            MetricData=[
                {
                    "MetricName": key,
                    "Value": 1,
                    "Unit": "Count",
                    "Dimensions": [{"Name": "Value", "Value": str(value)}],
                }
            ],
            Namespace=FEATURES_NAMESPACE,
        )

# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event: Dict[str, Any], context: Any | None = None) -> Dict[str, Any]:
    """Ponto de entrada para AWS Lambda."""
    logger.info("Received event: %s", event)

    # Extrai payload independente do modo de invocação
    if "body" in event:  # API Gateway
        body = json.loads(event.get("body", "{}"))
        payload = body.get("data", {})
    else:  # Invoke direto
        payload = event.get("data", {})

    try:
        features = prepare_payload(payload)
        prediction: float = float(MODEL.predict([features])[0])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error during prediction: %s", exc)
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(exc)}),
        }

    publish_metrics(payload, prediction)
    write_real_data_to_s3(payload, prediction)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(
            {
                "prediction": prediction,
                "version": MODEL_INFO["version"],
            }
        ),
    }

# ---------------------------------------------------------------------------
# Para debug local
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    example_event = {
        "data": {feature: 0 for feature in FEATURE_ORDER}  # Preencha conforme necessário
    }
    print(handler(example_event, None))
    