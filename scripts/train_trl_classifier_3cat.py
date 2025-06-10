# scripts/train_trl_classifier_3cat.py
import argparse
import numpy as np
import evaluate
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import classification_report, confusion_matrix
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    # 1. Load and Split Dataset
    logging.info("Loading and splitting dataset from %s...", args.csv_path)
    try:
        dataset = load_dataset("csv", data_files=args.csv_path, split="train")
    except Exception as e:
        logging.error(f"Failed to load dataset. Make sure the file exists and is a valid CSV: {e}")
        return

    # Check for required columns
    if "text" not in dataset.column_names or "trl_category" not in dataset.column_names:
        logging.error("CSV file must contain 'text' and 'trl_category' columns.")
        return

    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    logging.info(f"Training on {len(train_dataset)} samples, evaluating on {len(eval_dataset)} samples.")

    # 2. Load Tokenizer and Preprocess
    logging.info("Loading tokenizer '%s' and preprocessing data...", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    # Rename 'trl_category' to 'labels' for the Trainer
    train_dataset = train_dataset.rename_column("trl_category", "labels")
    eval_dataset = eval_dataset.rename_column("trl_category", "labels")
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    # 3. Define Metrics
    id2label = {0: "Early", 1: "Mid", 2: "Late"}
    
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        f1_metric = evaluate.load("f1")
        
        # Calculate overall F1 score
        f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")
        
        # Log detailed reports for analysis
        print("\n--- Classification Report ---")
        print(classification_report(labels, preds, target_names=id2label.values(), digits=3))
        print("\n--- Confusion Matrix ---")
        print(confusion_matrix(labels, preds))
        
        return {"f1": f1["f1"]}

    # 4. Load Model
    logging.info("Loading pre-trained model '%s' for sequence classification...", args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()}
    )

    # 5. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=50,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir='./logs/training_logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1, # Only keep the best checkpoint
    )

    # 6. Initialize Trainer and Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    logging.info("Starting model training...")
    trainer.train()

    # 7. Final Evaluation and Save
    logging.info("\n--- Final Evaluation on Test Set ---")
    final_metrics = trainer.evaluate()
    logging.info(f"Final Test Metrics: {final_metrics}")

    logging.info(f"\nTraining complete. Best model and tokenizer saved to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a SciBERT model for 3-category TRL classification.")
    parser.add_argument("--csv_path", type=str, default="data/labelled/trl_labels_3cat.csv", help="Path to the training CSV file.")
    parser.add_argument("--model_name", type=str, default="allenai/scibert_scivocab_uncased", help="Base model from Hugging Face.")
    parser.add_argument("--output_dir", type=str, default="models/trl_scibert_3cat", help="Directory to save the fine-tuned model.")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs.")
    
    args = parser.parse_args()
    main(args)