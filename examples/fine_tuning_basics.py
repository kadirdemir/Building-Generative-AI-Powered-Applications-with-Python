"""
Fine-Tuning Basics with Transformers
=====================================
Demonstrates how to fine-tune a pre-trained text classification model
on a custom dataset using the Hugging Face `transformers` library and PyTorch.

The example uses a small synthetic dataset of product reviews labeled as
positive or negative. It walks through:
  - Tokenising text with a pre-trained tokeniser
  - Wrapping data in a PyTorch Dataset
  - Loading a sequence-classification model with a fresh head
  - Running a minimal training loop with AdamW
  - Evaluating accuracy on a held-out test split
  - Saving the fine-tuned model and tokeniser to disk

Dependencies:
    pip install transformers torch scikit-learn

Note: Running on a CPU is intentional here so the script works out of the box
      without a GPU.  For larger datasets, switch `device` to "cuda" or "mps".
"""

from __future__ import annotations

import os
import random
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

POSITIVE_REVIEWS: List[str] = [
    "Absolutely love this product, works perfectly!",
    "Great quality and fast shipping, very happy.",
    "Exceeded my expectations — highly recommend.",
    "Solid build, easy setup, does exactly what it promises.",
    "Five stars without hesitation, will buy again.",
    "Impressive performance for the price.",
    "Customer support was helpful and the item arrived on time.",
    "Fantastic value, I use it every day.",
    "Well designed and durable, very satisfied.",
    "Best purchase I've made this year.",
]

NEGATIVE_REVIEWS: List[str] = [
    "Stopped working after two days, very disappointed.",
    "Poor quality, nothing like the photos.",
    "Arrived damaged and the return process was frustrating.",
    "Way overpriced for what you actually get.",
    "Instructions were confusing and the product is flimsy.",
    "Would not recommend — broke within a week.",
    "Terrible customer service, still waiting for a refund.",
    "Battery drains extremely fast, not as advertised.",
    "Looks cheap, feels cheap, complete waste of money.",
    "Missing parts and no response from the seller.",
]


def build_dataset() -> Tuple[List[str], List[int]]:
    """Return a shuffled list of (text, label) pairs.

    Labels: 1 = positive, 0 = negative.
    """
    texts = POSITIVE_REVIEWS + NEGATIVE_REVIEWS
    labels = [1] * len(POSITIVE_REVIEWS) + [0] * len(NEGATIVE_REVIEWS)
    combined = list(zip(texts, labels))
    random.seed(42)
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ReviewDataset(Dataset):
    """Tokenised review dataset for sequence classification."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader: DataLoader, optimiser, scheduler, device: str) -> float:
    """Run one full pass over the training data and return mean loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimiser.step()
        scheduler.step()
        optimiser.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader: DataLoader, device: str) -> float:
    """Return accuracy on the provided DataLoader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += len(batch["labels"])
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Configuration -------------------------------------------------------
    model_name = "distilbert-base-uncased"
    num_labels = 2
    max_length = 128
    batch_size = 4
    num_epochs = 3
    learning_rate = 2e-5
    output_dir = "fine_tuned_review_classifier"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Base model: {model_name}")

    # --- Data ----------------------------------------------------------------
    texts, labels = build_dataset()
    split = int(0.8 * len(texts))
    train_texts, test_texts = texts[:split], texts[split:]
    train_labels, test_labels = labels[:split], labels[split:]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = ReviewDataset(test_texts, test_labels, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # --- Model ---------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    # --- Training loop -------------------------------------------------------
    print("\nStarting training …")
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimiser, scheduler, device)
        accuracy = evaluate(model, test_loader, device)
        print(f"  Epoch {epoch}/{num_epochs}  loss={avg_loss:.4f}  test_acc={accuracy:.2%}")

    # --- Save ----------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nFine-tuned model saved to '{output_dir}/'")

    # --- Quick inference demo -----------------------------------------------
    sample_texts = [
        "This is an outstanding product, very pleased!",
        "Complete garbage, broke on the first day.",
    ]
    model.eval()
    for text in sample_texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        pred = logits.argmax(dim=-1).item()
        label_str = "positive" if pred == 1 else "negative"
        print(f"  Input : {text!r}")
        print(f"  Label : {label_str}\n")


if __name__ == "__main__":
    main()
