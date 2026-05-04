from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("models/model.joblib")
model = joblib.load(MODEL_PATH)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "Request body must be valid JSON"}), 400

        if "features" not in data:
            return jsonify({"error": "JSON must contain key 'features'"}), 400

        features = data["features"]

        if not isinstance(features, dict):
            return jsonify({"error": "'features' must be a JSON object"}), 400

        X_input = pd.DataFrame([features])

        prediction = int(model.predict(X_input)[0])
        probability = float(model.predict_proba(X_input)[0][1])

        return jsonify({
            "prediction": prediction,
            "probability": probability,
            "model_version": "v1"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)