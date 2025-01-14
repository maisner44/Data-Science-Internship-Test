## Named Entity Recognition (NER) for Mountain Names
This project demonstrates how to fine-tune a BERT-based model for detecting mountain names in text. It includes:

Data Preparation: A CSV file containing sentences and labeled entity spans (mountain mentions).
Model Training: Fine-tuning a pre-trained BERT model on the NER task.
Inference: Predicting mountain entities in new texts using the trained model.

Description of the Work
Dataset Creation: Prepared a CSV file (ner_mountain_dataset.csv) that holds text samples and entity annotations. Each annotation specifies the character offset range and label (e.g., MOUNTAIN) for a mountain name in the text.

1. How to run train.py

python train.py \
    --csv_file ner_mountain_dataset.csv \
    --model_name bert-base-cased \
    --output_dir ./bert_mountain_ner \
    --num_train_epochs 3 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --lr 2e-5

2. How to run inference.py

python inference.py \
    --model_dir ./bert_mountain_ner \
    --text "I have always dreamed of climbing K2 and Annapurna."
