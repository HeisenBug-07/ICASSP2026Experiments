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
  --train path/to/source.txt \
  --test path/to/target.txt \
  --valid path/to/valid.txt \
  --lang 12 \
