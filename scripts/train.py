import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
import mlflow
from azureml.core import Run, Dataset

class LIARDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["statement"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    preds, true, total_loss = [], [], 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            preds.extend(predictions.cpu().numpy())
            true.extend(batch["labels"].cpu().numpy())
    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average="macro")
    cm = confusion_matrix(true, preds)
    return total_loss / len(dataloader), acc, f1, cm

def main(args):
    mlflow.start_run()
    mlflow.log_params({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_len": args.max_len
    })
    
    run = Run.get_context()
    ws = run.experiment.workspace
    train_ds = Dataset.get_by_name(ws, args.train_dataset)
    val_ds = Dataset.get_by_name(ws, args.val_dataset)
    train_path = train_ds.download(target_path='data/train', overwrite=True)
    val_path = val_ds.download(target_path='data/valid', overwrite=True)
    train_df = pd.read_csv(train_path[0], sep="\t")
    val_df = pd.read_csv(val_path[0], sep="\t")
    
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_dataset = LIARDataset(train_df, tokenizer, max_len=args.max_len)
    val_dataset = LIARDataset(val_df, tokenizer, max_len=args.max_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=6
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    best_f1 = 0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, acc, f1, cm = evaluate(model, val_loader, device)
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Accuracy:   {acc:.4f}")
        print(f"F1 Score:   {f1:.4f}")
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": acc,
            "val_f1": f1
        }, step=epoch)
        if f1 > best_f1:
            best_f1 = f1
            output_dir = "outputs/best_model"
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            mlflow.log_artifacts(output_dir)
    mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", type=str, default="Liar_train_clean")
    parser.add_argument("--val_dataset", type=str, default="Liar_valid_clean")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()
    main(args)
