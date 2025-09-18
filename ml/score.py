import json, os, glob, joblib, logging
import numpy as np

logging.basicConfig(level=logging.INFO)

def init():
    global model

    model_root = os.getenv("AZUREML_MODEL_DIR", ".")
    matches = glob.glob(os.path.join(model_root, "**", "model.joblib"), recursive=True)
    if not matches:
        raise FileNotFoundError(f"model.joblib not found under {model_root}")
    model_path = matches[0]
    logging.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    logging.info("Model loaded")

def run(raw_data):
    try:
        # Accept either a JSON string or already-parsed dict
        payload = json.loads(raw_data) if isinstance(raw_data, (str, bytes)) else raw_data
        data = np.array(payload["data"])
        preds = model.predict(data).tolist()
        logging.info({"predictions": preds})
        return {"predictions": preds}
    except Exception as e:
        logging.exception("Scoring error")
        return {"error": str(e)}
