"Predict one delivery, from the command line."

import pandas as pd
from pathlib import Path

from delivery import load_model


def main():
    model_path = Path(__file__).parent / "model.joblib"
    model = load_model(model_path)

    row = pd.DataFrame([{
        "distance_km": 7.0,
        "prep_time_min": 25,
        "traffic_level": 3,
        "rain": 0
    }])

    prediction = model.predict(row)[0]

    print(f"PREDICTION: {prediction:.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
