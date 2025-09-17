# Domain-Adversarial Spoken Language Identification

This repository contains a **Conformer-based spoken language identification (SLID) model** with **domain adaptation** using a **Gradient Reversal Layer (GRL)**.  
The model is trained to classify languages from a source domain while learning **domain-invariant representations** by adversarially training against a domain classifier.

---

## Features
- **Phoneme posteriors** input features.
- **Conformer backbone** for feature modeling.  
- **Domain adaptation** via Gradient Reversal Layer (GRL).  
- **Cross-entropy losses** for language and domain classification.  
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
