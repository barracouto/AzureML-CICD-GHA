import json, os, joblib, logging
logging.basicConfig(level=logging.INFO)

def init():
    global model
    path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "model.joblib")
    model = joblib.load(path)
    logging.info("Model loaded")

def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        preds = model.predict(data).tolist()
        logging.info({"predictions": preds})
        return {"predictions": preds}
    except Exception as e:
        logging.exception("Scoring error")
        return {"error": str(e)}
# update to deploy