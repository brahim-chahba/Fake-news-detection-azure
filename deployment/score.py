import os
import json
from pathlib import Path
from inference import FakeNewsClassifier

_model = None

def init():
    global _model
    # Azure injects the registered model into AZUREML_MODEL_DIR
    model_dir = os.getenv("AZUREML_MODEL_DIR", None)
    if model_dir is None:
        # Fallback if running locally
        model_dir = os.path.join(os.getcwd(), "model")

    # Detect if model files are in a single subfolder
    subfolders = [f for f in Path(model_dir).iterdir() if f.is_dir()]
    if len(subfolders) == 1:
        model_dir = str(subfolders[0])

    _model = FakeNewsClassifier(model_dir=model_dir, device="cpu", max_length=128)

def run(raw_data):
    """
    Accepts:
      - raw_data: JSON string like {"input":"is this true"} OR
                  {"input":["text1", "text2"]} OR
                  array of objects [{"input":"t1"}, {"input":"t2"}]
    Returns:
      - JSON serializable response with prediction(s)
    """
    try:
        data = json.loads(raw_data)
    except Exception:
        return {"error": "Invalid JSON"}

    if isinstance(data, dict) and "input" in data:
        if isinstance(data["input"], str):
            return _model.predict_single(data["input"])
        elif isinstance(data["input"], list):
            return _model.predict_batch(data["input"])

    if isinstance(data, list):
        texts = []
        for item in data:
            if isinstance(item, dict) and "input" in item:
                texts.append(item["input"])
            else:
                return {"error": "List items must be objects with 'input' key"}
        return _model.predict_batch(texts)

    return {"error": "Unrecognized input format. Use {'input':'text'} or {'input':['t1','t2']} or [{'input':'t1'},...]"}
