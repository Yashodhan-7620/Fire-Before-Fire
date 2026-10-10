"""
SHAP-based explainability for the forecaster models, used to turn a
"threshold crossed" prediction into a human-readable recommendation
("what aspect to focus on") instead of just a raw number.

Both the forecaster (RandomForestRegressor) and the anomaly scorer
(IsolationForest) are tree ensembles, so shap.TreeExplainer works for
either without extra configuration.
"""
import shap

# Human-friendly guidance keyed by the *base* sensor parameter (the
# feature-column prefix before _mean/_std/_max), used when that
# parameter's engineered feature is the top SHAP contributor driving a
# threshold-crossing prediction.
RECOMMENDATIONS = {
    "temperature_c": (
        "Inspect cooling/ventilation and nearby combustible material -- "
        "rising temperature is the dominant driver of this prediction."
    ),
    "voltage_v": "Check the incoming supply and wiring for voltage instability on this node.",
    "current_a": (
        "Inspect wiring, connectors and the connected load -- unusually high "
        "current draw is the dominant driver of this prediction."
    ),
    "power_w": (
        "Review connected appliances/load on this circuit -- power draw is the "
        "dominant driver of this prediction."
    ),
}


def explain_row(model, x_row):
    """Returns [(feature_name, shap_value), ...] sorted by |impact|,
    for a single-row feature DataFrame `x_row`."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_row)
    values = shap_values[0] if hasattr(shap_values, "__len__") else shap_values
    contributions = list(zip(x_row.columns, [float(v) for v in values]))
    contributions.sort(key=lambda t: abs(t[1]), reverse=True)
    return contributions


def _base_param(feature_name: str) -> str:
    for suffix in ("_mean", "_std", "_max", "_min"):
        if feature_name.endswith(suffix):
            return feature_name[: -len(suffix)]
    return feature_name


def build_recommendation(contributions_by_param: dict) -> str | None:
    """Given {param: [(feature, shap_value), ...]} for the parameters
    that crossed their threshold, pick the single strongest SHAP
    contributor overall and translate it into guidance."""
    best_feature, best_value = None, 0.0
    for contributions in contributions_by_param.values():
        if not contributions:
            continue
        feature, value = contributions[0]
        if abs(value) >= abs(best_value):
            best_feature, best_value = feature, value
    if best_feature is None:
        return None
    param = _base_param(best_feature)
    return RECOMMENDATIONS.get(
        param, f"Investigate {param} on this node -- it is the dominant driver of the alert.",
    )
