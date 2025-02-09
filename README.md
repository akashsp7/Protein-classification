# Protein Subcellular Localization Predictor

A deep learning model for predicting protein subcellular localization using the ESM2 language model. This project aims to classify proteins into 12 different subcellular locations based on their amino acid sequences.

## Overview

The model leverages the power of Meta AI's ESM2 (Evolutionary Scale Modeling) transformer model, specifically the ESM2-3B-UR50D variant, to understand protein sequences and predict their subcellular locations. The model achieves impressive performance with:

- Top-1 Accuracy: 92.14%
- Top-3 Accuracy: 97.06%
- Top-5 Accuracy: 98.38%

### Supported Subcellular Locations

The model can predict the following 12 subcellular locations:
- Cell Junction
- Cell Membrane
- Cell Projection
- Cytoplasm
- Endoplasmic Reticulum
- Golgi Apparatus
- Lipid-anchor
- Mitochondrion
- Nucleus
- Secreted
- Synapse
- Vesicular

## Project Structure

```
├── config.py           # Configuration settings and paths
├── evaluate.py         # Model evaluation script
├── random_predictions.py # Script for making random predictions
├── scripts.py          # Core functionality and helper classes
└── train.py           # Model training script
```

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Pandas
- Scikit-learn
- tqdm

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/protein-localization-predictor.git
cd protein-localization-predictor
```

2. Install the required packages:
```bash
pip install torch transformers pandas scikit-learn tqdm
```

## Usage

### Training

To train the model:

```bash
python train.py
```

The training script includes:
- Gradient checkpointing for memory efficiency
- Mixed precision training (FP16)
- Early stopping
- Linear learning rate scheduler
- Gradient accumulation for effective batch size of 16

### Evaluation

To evaluate the model:

```bash
python evaluate.py
```

### Making Predictions

To make predictions on random test samples:

```bash
python random_predictions.py --num_predictions 10
```

## Model Architecture

- Base Model: ESM2-3B-UR50D
- Fine-tuned with a classification head
- Training optimizations:
  - Gradient checkpointing
  - Adafactor optimizer
  - FP16 precision
  - Batch size: 4 with gradient accumulation steps of 4

## Dataset

The model is trained on a protein sequence dataset with subcellular location annotations. The data preprocessing includes:
- Removal of uncategorized samples
- Handling of special cases and location variants
- Priority-based categorization for proteins with multiple locations

The model is trained on human protein sequences obtained from UniProt using their REST API. The dataset was collected with the following criteria:

- Organism: Human (Taxonomy ID: 9606)
- Review Status: Swiss-Prot (reviewed)
- Sequence Length: 80-500 amino acids
- Fields Retrieved: 
  - Accession
  - Sequence
  - Subcellular Location

### Data Collection

The data was pulled directly from UniProt's REST API using the following Python code:

```python
import requests
import pandas as pd
from io import BytesIO

# UniProt REST API query URL with filters
query_url = "https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Csequence%2Ccc_subcellular_location&format=tsv&query=%28%28organism_id%3A9606%29%20AND%20%28reviewed%3Atrue%29%20AND%20%28length%3A%5B80%20TO%20500%5D%29%29"

# Fetch and load the data
uniprot_request = requests.get(query_url)
bio = BytesIO(uniprot_request.content)
df = pd.read_csv(bio, compression='gzip', sep='\t')
```

## Performance

The model demonstrates strong performance across different metrics:
- Top-1 Accuracy: 92.14% (exact prediction)
- Top-3 Accuracy: 97.06% (correct location in top 3 predictions)
- Top-5 Accuracy: 98.38% (correct location in top 5 predictions)
- Top-10 Accuracy: 99.70% (correct location in top 10 predictions)

## Limitations

- Requires significant computational resources (trained on A100 GPU)
- Currently only supports single-label classification
- Model size is approximately 3B parameters

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Meta AI for the ESM2 model
- Uniprot for the dataset
