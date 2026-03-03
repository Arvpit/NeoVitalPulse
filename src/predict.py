import os
import joblib
import pandas as pd
import shap
from src.utils import risk_category
from src.preprocess import preprocess_input
# =============================
# RESOLVE MODEL PATH
# =============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_rf_model.pkl")

# =============================
# LOAD MODEL
# =============================
model = joblib.load(MODEL_PATH)

# =============================
# CREATE SHAP EXPLAINER
# =============================
explainer = shap.TreeExplainer(model)


def predict_heart_disease(input_data: dict):
    """
    Predict heart disease risk with explanation
    """

    # Convert to DataFrame
    df = preprocess_input(input_data)

    # Predict probability
    prob = model.predict_proba(df)[0][1]

    # Risk level
    risk = risk_category(prob)

    # SHAP explanation
    shap_output = explainer.shap_values(df)

    # Binary classification handling
    if isinstance(shap_output, list):
        shap_values = shap_output[1][0]
    else:
        shap_values = shap_output[0]

    # Create contribution dataframe
    contrib_df = pd.DataFrame({
        "Feature": df.columns,
        "SHAP Value": shap_values
    })

    # Sort by impact
    contrib_df = contrib_df.reindex(
        contrib_df["SHAP Value"].abs().sort_values(ascending=False).index
    )

    # Create top 3 with values
    top_positive_df = contrib_df[contrib_df["SHAP Value"] > 0].head(3)
    top_negative_df = contrib_df[contrib_df["SHAP Value"] < 0].head(3)

    top_positive = [
    {"feature": row["Feature"], "impact": float(row["SHAP Value"])}
    for _, row in top_positive_df.iterrows()
    ]

    top_negative = [
    {"feature": row["Feature"], "impact": float(row["SHAP Value"])}
    for _, row in top_negative_df.iterrows()
    ]   
        # Add full shap contributions (sorted)
    full_contrib = contrib_df.sort_values(by="SHAP Value", key=abs, ascending=False)

    full_contrib_list = [
        {"feature": row["Feature"], "impact": float(row["SHAP Value"])}
        for _, row in full_contrib.iterrows()
    ]
    return {
        "probability": float(prob),
        "risk_level": risk,
        "top_risk_factors": top_positive,
        "top_protective_factors": top_negative,
        "all_contributions": full_contrib_list
    }