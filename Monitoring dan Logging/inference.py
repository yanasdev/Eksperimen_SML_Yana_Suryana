import json
import os
from pathlib import Path

import pandas as pd
import requests


MODEL_URL = os.getenv("MODEL_URL", "http://127.0.0.1:5000/invocations")


def predict(sample: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    response = requests.post(MODEL_URL, headers=headers, data=json.dumps(sample))
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    sample = {
        "dataframe_split": {
            "columns": ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF"],
            "data": [[7, 1710, 2, 856]],
        }
    }
    print(predict(sample))
