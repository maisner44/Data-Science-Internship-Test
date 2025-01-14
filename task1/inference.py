import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for a fine-tuned BERT NER model.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./bert_mountain_ner",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Mount Everest and K2 are both on my bucket list.",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    
    # Create a pipeline for token classification
    nlp = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    
    # Inference on the user-provided text
    print(f"Input text: {args.text}\n")
    entities = nlp(args.text)
    
    print("Entities found:")
    for ent in entities:
        print(f"  - Text: {ent['word']}, Label: {ent['entity_group']}, Score: {ent['score']:.4f}")
    
    print("\nInference complete.")

if __name__ == "__main__":
    main()
