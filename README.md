# Toxicity Prediction Project

A machine learning project for predicting molecular toxicity using various molecular descriptors and deep learning approaches.

## Overview

This project implements machine learning models to predict the toxicity of chemical compounds based on their SMILES representations. The system includes data preprocessing, feature extraction using molecular fingerprints (ECFP4), and multiple modeling approaches including Random Forest and Neural Networks.

## Project Structure

```
toxicity_prediction/
├── notebooks/              # Jupyter notebooks
│   ├── experiments.ipynb   # Experimental analysis
├── scripts/                # Python scripts
│   ├── main.py            # Main execution script
│   ├── utils.py           # Utility functions
│   ├── embeddings.py      # Embedding generation
├── environment.yml        # Conda environment specification
└── README.md             # This file
```

## Features

- **Molecular Data Processing**: SMILES validation, salt removal, tautomer canonicalization
- **Feature Extraction**: ECFP4 molecular fingerprints, ChemBERTa embeddings
- **Machine Learning Models**: Random Forest, Neural Networks
- **Data Balancing**: Oversampling/undersampling techniques for imbalanced datasets
- **Scaffold-based Splitting**: Ensures proper train/test splits based on molecular scaffolds
- **Comprehensive Evaluation**: Multiple metrics including precision, recall, F1-score, and ROC-AUC

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/GavrilenkoA/Toxicity-Prediction.git
   cd toxicity_prediction
   ```

2. **Create conda environment**:
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment**:
   ```bash
   conda activate toxicity_prediction
   ```

## Dependencies

The project requires the following key dependencies:
- Python 3.11.13
- RDKit 2025.03.4 (for molecular processing)
- PyTorch 2.7.1 (for neural networks)
- scikit-learn 1.7.0 (for traditional ML)
- pandas 2.2.3 (for data manipulation)
- Other dependencies listed in `environment.yml`

## Usage

### Basic Usage

Run the main prediction pipeline:

```bash
python scripts/main.py --input_path data/tox_dataset.csv --output_path data/output.csv
```

### Data Preprocessing

The system automatically performs the following preprocessing steps:

1. **SMILES Validation**: Filters out invalid SMILES strings
2. **Salt Removal**: Removes salts from molecules using RDKit
3. **Tautomer Canonicalization**: Standardizes tautomeric forms
4. **Organic Filtering**: Keeps only organic molecules (containing carbon)
5. **Ambiguity Removal**: Removes molecules with conflicting toxicity labels

### Model Training

The default pipeline uses Random Forest with the following configuration:
- 500 estimators
- ECFP4 fingerprints (2048 bits, radius 2)
- Class balancing via oversampling
- Random train/test splitting

### Advanced Usage

For custom model training and evaluation, use the notebooks:

1. **Experiments** (`notebooks/experiments.ipynb`): Comprehensive model comparison
2. **Embeddings** (`notebooks/embeddings.ipynb`): ChemBERTa embedding analysis

## Data Format

### Input Data
The input CSV should contain:
- `SMILES`: Molecular structure in SMILES format
- `toxicity`: Binary toxicity labels (0 = non-toxic, 1 = toxic)

### Output Data
The output CSV contains:
- Original columns from input
- `y_pred`: Predicted toxicity class
- `y_proba`: Prediction probability

## Key Functions

### Data Processing
- `filter_and_log()`: Comprehensive molecular filtering with logging
- `remove_ambiguous_smiles()`: Removes molecules with conflicting labels
- `is_valid_organic_smiles()`: Validates organic SMILES strings

### Feature Extraction
- `featurize_ecfp4()`: Generates ECFP4 molecular fingerprints

### Model Training
- `balance_classes()`: Handles class imbalance
- `train_test_split()`: Scaffold-based data splitting
- `compute_metrics()`: Calculates comprehensive evaluation metrics

## Evaluation Metrics

The system evaluates models using:
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Sensitivity**: True negative rate
