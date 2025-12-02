import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from pathlib import Path

class FakeNewsClassifier:
    def __init__(self, model_dir, device=None, max_length=128):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        # Optional: confirm the directory exists
        model_path = Path(model_dir)
        if not model_path.exists():
            raise ValueError(f"Model directory {model_dir} does not exist")

        # Load tokenizer and model
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir, local_files_only=True)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    def predict_single(self, text):
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy().tolist()[0]
            pred = int(torch.argmax(logits, dim=-1).cpu().item())
        return {"prediction": pred, "label": "Real" if pred == 1 else "Fake", "probabilities": probs}

    def predict_batch(self, texts):
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = self.model(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy().tolist()
            preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        results = []
        for p, pr in zip(preds, probs):
            results.append({"prediction": int(p), "label": "Real" if p == 1 else "Fake", "probabilities": pr})
        return results
