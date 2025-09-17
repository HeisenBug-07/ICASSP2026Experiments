# Spoken Language Identification with Conformer and Phoneme

This repository contains a **Conformer-based spoken language identification (SLID) model**.  

---

## Features
- **Phoneme posteriors** input features.
- **Conformer backbone** for feature modeling.  
- **Cross-entropy losses** for language.  
- Evaluation with **Accuracy, Balanced Accuracy, EER, and Cavg**.

---

## Usage

### Training
```bash
python train_conformer.py \
  --train path/to/train_data.txt \
  --test path/to/test_data.txt \
  --lang 12 \
