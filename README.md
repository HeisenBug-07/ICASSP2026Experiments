# Domain-Adversarial Training for Robust Language Identification

This repository contains implementations of experiments evaluating **phoneme-level features** combined with **adversarial domain adaptation** (via Gradient Reversal Layers) for multilingual **spoken language identification (LID)**.  
Our framework leverages Conformer-based encoders and adversarial training strategies to improve **cross-domain robustness** and **low-resource recognition**.

---

## Table of Contents
- [Overview](#overview)  
- [Experiments](#experiments)  
- [Setup and Installation](#setup-and-installation)  
- [Data Preparation](#data-preparation)  
- [Training](#training)  
- [Evaluation](#evaluation)  
- [Results](#results)  
- [Configurations](#configurations)  
- [Citation](#citation)  
- [Contact](#contact)  

---

## 🔎 Overview
The codebase supports the following key experiments:

- **Experiment 1**: Baseline (no domain adaptation) with Ekstep training data.  
- **Experiment 2**: Baseline with an alternative architecture (e.g., Vision Transformer).  
- **Experiment 3**: Domain adaptation to YouTube domain using Conformer + GRL.  
- **Experiment 4**: Joint training with Ekstep + Vaani; adaptation to YouTube.  
- **Experiment 5**: Multi-source training without adaptation.  
- **Experiment 6**: Multi-target adversarial training (best-performing).  

All models are trained on **Indian-language datasets** (Ekstep, IIT Mandi, Vaani, IndicVoice), focusing on **low-resource** and **multi-domain** conditions.

---

## 🧪 Experiments

| Exp | Description                                | Domain Adaptation | Source Data     | Target Data       | Notes                                   |
|-----|--------------------------------------------|-------------------|-----------------|------------------|-----------------------------------------|
| 1   | Baseline (no adaptive training)            | No                | Ekstep          | Ekstep test       | In-domain benchmark                     |
| 2   | Baseline (ViT architecture)                | No                | Ekstep          | Ekstep test       | Architecture comparison                 |
| 3   | YouTube domain adaptation                  | Yes (GRL)         | Ekstep          | IIT Mandi (YT)    | Explicit domain adaptation               |
| 4   | Joint training + adaptation                | Yes (GRL)         | Ekstep + Vaani  | IIT Mandi (YT)    | Dataset augmentation + adaptation        |
| 5   | Multi-source training only                 | No                | Ekstep + Vaani  | IIT Mandi (YT)    | Effect of data diversity alone           |
| 6   | Multi-target adversarial training (best)   | Yes (GRL)         | Ekstep + Vaani  | Multiple targets  | Best overall generalization              |

---

## 📂 Data Preparation

Collect and preprocess Indian-language speech datasets:

- **Ekstep** (md, tv)  
- **IIT Mandi** (rs, yt)  
- **Vaani**  
- **IndicVoice**

### Feature Extraction
- Preferred: **Phoneme-level features**  
- Alternatives: **MFCCs, spectrograms, or embeddings**  
- Use the provided **`phoneme_feature_extraction.py`** script to extract and store features.  

### Dataset Splits
- Create **train / validation / test** splits for each experiment.  
- Ensure stratified sampling where possible to balance class distributions.  

### Manifest Files
- Prepare manifest files in `.csv` or `.txt` format containing:  
  - `audio_path` → path to the audio file  
  - `label` → language ID  

---

## 📊 Results

The table below summarizes the performance of different experiments on **seen** and **unseen** test sets.  
**DA (GRL)** = Domain Adaptation using Gradient Reversal Layer.

| Experiment   | DA (GRL) | Source (train)              | Target (adapted)                | Ekstep (seen-test) | IITM-YT (unseen) | IndicVoice (unseen) |
|--------------|----------|-----------------------------|---------------------------------|--------------------|------------------|---------------------|
| Experiment 1 | No       | ekstep-train               | —                               | 91%                | 83%              | 49%                 |
| Experiment 2 | No       | ekstep-train + vaani-train | —                               | 89%                | 74%              | 44%                 |
| Experiment 3 | Yes      | ekstep-train               | iitmandi_yT                     | 85%                | 83%              | 47%                 |
| Experiment 4 | Yes      | ekstep-train + vaani-train | iitmandi_yT                     | 81%                | 80%              | 51%                 |
| Experiment 5 | No       | ekstep-train + vaani-train | —                               | 90%                | 85%              | **57%**             |
| Experiment 6 | Yes      | ekstep-train               | iitmandi_yT + vaani-seen-train  | **93%**            | **86%**          | 52%                 |

➡️ **Best overall performance**:  
- **Seen-test (Ekstep):** 93% (Experiment 6)  
- **Unseen IITM-YT:** 86% (Experiment 6)  
- **Unseen IndicVoice:** 57% (Experiment 5)

  ---

  ## 👉 For instructions on running **individual models**, please check their respective **`README.md`** files
