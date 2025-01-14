import argparse
import ast
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
import torch

# Parse Command-Line Arguments

def parse_args():
    parser = argparse.ArgumentParser(description="Train a BERT-based NER model.")
    parser.add_argument(
        "--csv_file",
        type=str,
        default="ner_mountain_dataset.csv",
        help="Path to the CSV file with text and entity annotations."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-cased",
        help="Pretrained model name or path (Hugging Face Hub or local)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./bert_ner_model",
        help="Directory to save the fine-tuned model."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Training batch size."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Evaluation/validation batch size."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate."
    )
    args = parser.parse_args()
    return args

# Data Loading & Preprocessing

def read_csv_data(csv_file):

    df = pd.read_csv(csv_file)
    df["entities"] = df["entities"].apply(ast.literal_eval)

    def convert_tuples_to_dicts(tuple_list):
        dict_list = []
        for (start, end, label) in tuple_list:
            start = int(start)
            end = int(end)
            dict_list.append({
                "start": start,
                "end": end,
                "label": label
            })
        return dict_list
    
    df["entities"] = df["entities"].apply(convert_tuples_to_dicts)
    
    return df

def df_to_hf_dataset(df):

    data_dict = {
        "text": df["text"].tolist(),
        "entities": df["entities"].tolist()
    }
    dataset = Dataset.from_dict(data_dict)
    return dataset

def train_val_split(dataset, val_ratio=0.2, seed=42):
    """
    Splits the dataset into train and validation subsets.
    """
    dataset = dataset.train_test_split(test_size=val_ratio, seed=seed)
    dataset_dict = DatasetDict({
        "train": dataset["train"],
        "validation": dataset["test"]
    })
    return dataset_dict


# Label Scheme & Token Alignment

label_list = ["O", "B-MOUNTAIN", "I-MOUNTAIN"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

def tokenize_and_align_labels(examples, tokenizer, max_length=128):
    tokenized_inputs = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True
    )
    
    all_entities = examples["entities"]
    offset_mappings = tokenized_inputs["offset_mapping"]
    labels = []
    
    for i, offsets in enumerate(offset_mappings):
        entity_spans = all_entities[i]
        label_ids = ["O"] * len(offsets)
        
        for entity in entity_spans:
            start_char = entity["start"]
            end_char = entity["end"]
            ner_label = entity["label"]
            
            for idx, (offset_start, offset_end) in enumerate(offsets):
                if offset_start < end_char and offset_end > start_char:
                    if label_ids[idx] == "O":
                        label_ids[idx] = f"B-{ner_label}"
                    else:
                        label_ids[idx] = f"I-{ner_label}"
        
        label_ids = [
            label2id[label] if label in label2id else label2id["O"] 
            for label in label_ids
        ]
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs

# 4. Training Pipeline

def main():
    args = parse_args()
    
    df = read_csv_data(args.csv_file)
    hf_dataset = df_to_hf_dataset(df)
    dataset_dict = train_val_split(hf_dataset, val_ratio=0.2)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    def process_examples(examples):
        return tokenize_and_align_labels(examples, tokenizer)
    
    dataset_dict = dataset_dict.map(process_examples, batched=True)
    
    dataset_dict.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        logging_steps=50,
        logging_dir=f"{args.output_dir}/logs",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
    )
    
    from seqeval.metrics import precision_score, recall_score, f1_score
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        true_labels = [[id2label[l.item()] for l in label_row] for label_row in labels]
        pred_labels = [[id2label[pred] for pred in pred_row] for pred_row in predictions]
        
        precision = precision_score(true_labels, pred_labels, mode="strict")
        recall = recall_score(true_labels, pred_labels, mode="strict")
        f1 = f1_score(true_labels, pred_labels, mode="strict")
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
