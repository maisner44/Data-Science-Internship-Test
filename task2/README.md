## Description of the Work

**Preprocessing**  
Dataset was downloaded form kaggle in the directory 'data/' contains multiple subfolders, each corresponding to a Sentinel-2 

**Model Architecture**  
**Feature Extractor** CNN transforms each patch into a descriptor vector.  
**Siamese Head** compares two embeddings (e.g., via absolute difference) and outputs a smaller feature.

**Training**  
During training, a loss function (e.g., MSE or contrastive) encourages similar patches to have similar embeddings.

**Inference**  
Loads the trained model, processes two patches, and returns a distance measure indicating similarity.


1. How to Run train.py

python train.py \
  --data_dir "data/" \
  --batch_size 4 \
  --epochs 5 \
  --lr 1e-4 \
  --patch_size 256 \
  --device cuda

2. How to Run inference.py

python inference.py \
    --model_path siamese_model.pth \
    --img1 "data/S2A_MSIL1C..._B01.jp2" \
    --img2 "data/S2A_MSIL1C..._B02.jp2" \
    --patch_size 256 \
    --device cpu
