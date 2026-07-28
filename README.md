# Disease-Prediction-System-using-BERT-NLP-
A disease prediction system powered by the BERT Base Uncased transformer model that classifies diseases from patient symptom descriptions using PyTorch and Hugging Face Transformers.

# Disease Prediction using BERT Base Uncased

## Overview

This project implements a **Natural Language Processing (NLP)** disease prediction system using the **BERT Base Uncased** transformer model. The model is fine-tuned on symptom descriptions to accurately classify diseases from free-text patient inputs.

Instead of relying on manually engineered features, the system leverages transfer learning from a pretrained BERT model, enabling it to understand the context and relationships between symptoms for multi-class disease classification.

---

## Features

* Fine-tuned **BERT Base Uncased** for disease classification
* Multi-class disease prediction from symptom descriptions
* Automatic text tokenization using Hugging Face Tokenizer
* End-to-end training and evaluation pipeline
* Model saving and loading for inference
* Prediction probabilities for each disease class
* GPU support through PyTorch

---

## Model Architecture

```text
Patient Symptoms
        │
        ▼
BERT Tokenizer
        │
        ▼
WordPiece Tokens
        │
        ▼
BERT Base Uncased
(12 Transformer Encoder Layers)
        │
        ▼
[CLS] Sentence Representation
        │
        ▼
Dropout
        │
        ▼
Linear Classification Layer
        │
        ▼
Softmax
        │
        ▼
Predicted Disease
```

---

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Scikit-learn
* Pandas
* NumPy
* Matplotlib

---

## Project Structure

```text
├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── models/
│   ├── saved_model/
│   └── tokenizer/
│
├── notebooks/
│
├── train.py
├── evaluate.py
├── predict.py
├── dataset.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/disease-prediction-bert.git

cd disease-prediction-bert
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Dataset

The dataset consists of:

* Symptom descriptions (input text)
* Disease labels (target classes)

Example:

| Symptoms                                | Disease       |
| --------------------------------------- | ------------- |
| Fever, cough, sore throat               | Influenza     |
| Frequent urination and excessive thirst | Diabetes      |
| Chest pain and shortness of breath      | Heart Disease |

---

## Training

Run:

```bash
python train.py
```

Typical training configuration:

```python
Model: bert-base-uncased
Batch Size: 16
Learning Rate: 2e-5
Epochs: 5
Optimizer: AdamW
Loss Function: CrossEntropyLoss
```

---

## Evaluation

Run:

```bash
python evaluate.py
```

Common evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## Prediction

Run:

```bash
python predict.py
```

Example input:

```text
Patient has fever, cough, headache, and fatigue.
```

Example output:

```text
Predicted Disease:
Influenza

Confidence:
97.4%
```

---

## Fine-Tuning Process

The pretrained **BERT Base Uncased** model is fine-tuned for disease classification by:

1. Tokenizing symptom descriptions using the BERT tokenizer.
2. Passing tokenized inputs through the pretrained BERT encoder.
3. Extracting the contextual representation of the `[CLS]` token.
4. Feeding the representation into a linear classification layer.
5. Computing the CrossEntropy loss.
6. Updating model parameters using the AdamW optimizer through backpropagation.

This process allows the model to adapt its pretrained language understanding to the medical disease classification task.

---

## Results

The fine-tuned model learns contextual relationships between symptoms rather than relying solely on keywords, resulting in improved disease prediction performance compared to traditional machine learning methods.

---

## Future Improvements

* Deploy as a web application using FastAPI or Flask.
* Add explainable AI (XAI) visualizations.
* Support multilingual symptom descriptions.
* Integrate medical knowledge bases.
* Expand to larger clinical datasets.

---

## Acknowledgements

* Hugging Face Transformers
* PyTorch
* Google Research for the original BERT model
* Scikit-learn
