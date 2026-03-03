import pandas as pd
import json
import os

# Load feature list
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_columns.json")

with open(FEATURE_PATH, "r") as f:
    feature_columns = json.load(f)


def preprocess_input(input_data: dict):
    """
    Convert raw user input into model-ready dataframe
    """

    df = pd.DataFrame([input_data])

    # Binary mapping
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

    df['fasting_blood_sugar'] = df['fasting_blood_sugar'].map({
        'Lower than 120 mg/ml': 0,
        'Greater than 120 mg/ml': 1
    })

    df['exercise_induced_angina'] = df['exercise_induced_angina'].map({
        'No': 0,
        'Yes': 1
    })

    df['vessels_colored_by_flourosopy'] = df['vessels_colored_by_flourosopy'].map({
        'Zero': 0,
        'One': 1,
        'Two': 2,
        'Three': 3,
        'Four': 4
    })

    # One-hot encoding
    df = pd.get_dummies(
        df,
        columns=[
            'chest_pain_type',
            'rest_ecg',
            'slope',
            'thalassemia'
        ],
        drop_first=True
    )

    # Ensure all required columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    df = df[feature_columns]

    return df