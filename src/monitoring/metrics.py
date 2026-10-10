"""
Prometheus instrumentation for the web dashboard. A dedicated registry
(rather than the global default) keeps this importable/testable
without colliding with other collectors in the same process.
"""
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

DRIFT_CHECKS = Counter(
    "drift_checks_total", "Total number of drift checks performed", registry=REGISTRY,
)
DRIFT_DETECTED = Counter(
    "drift_events_total", "Total number of times drift was detected", registry=REGISTRY,
)
RETRAIN_RUNS = Counter(
    "retrain_runs_total", "Total number of retrain runs", ["trigger"], registry=REGISTRY,
)
RETRAIN_FAILURES = Counter(
    "retrain_failures_total", "Total number of failed retrain runs", registry=REGISTRY,
)
BATCH_PREDICTIONS = Counter(
    "batch_predictions_total", "Total number of batch inference rows served", registry=REGISTRY,
)
REALTIME_PREDICTIONS = Counter(
    "realtime_predictions_total", "Total number of real-time inference requests served",
    registry=REGISTRY,
)
EMERGENCY_ALERTS = Counter(
    "emergency_alerts_total", "Total number of threshold-crossing emergencies raised",
    ["mode"], registry=REGISTRY,
)
LAST_PREDICTED_VALUE = Gauge(
    "last_predicted_value", "Most recent predicted value per node/parameter/mode",
    ["node_id", "parameter", "mode"], registry=REGISTRY,
)
INFERENCE_LATENCY_SECONDS = Histogram(
    "inference_latency_seconds", "Inference latency in seconds", ["mode"], registry=REGISTRY,
)
MODEL_LAST_TRAINED_TIMESTAMP = Gauge(
    "model_last_trained_timestamp", "Unix timestamp of the last successful retrain",
    registry=REGISTRY,
)
