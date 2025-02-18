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
├── config.py              # Configuration settings and paths
├── evaluate.py            # Model evaluation script
├── random_predictions.py  # Script for making random predictions
├── scripts.py             # Core functionality and helper classes
├── train.py               # Model training script
├── datasets               # Folder to save train/test dataset (torch datasets)
├── Model                  # The fine-tuned model is saved here
└── protein_folding.csv    # Our data in csv format
```
# Random Predictions

Here are some example predictions from the model showing its ability to predict protein subcellular locations from sequences:

## Example 1: Nuclear Protein
**Entry ID:** Q02363  
**Sequence (truncated):** `MKAFSPVRSVRKNSLSDHSLGISRSKTPVDDPMSLLYNMNDCYSKLKELV...`

### Results:
- **True Location:** Nucleus
- **Predicted Location:** Nucleus
- **Confidence:** 100.00%

**Top 3 Predictions:**
1. Nucleus (100.00%)
2. Endoplasmic reticulum (0.00%)
3. Cell projection (0.00%)

## Example 2: Membrane Protein
**Entry ID:** Q8NGE3  
**Sequence (truncated):** `MAGENHTTLPEFLLLGFSDLKALQGPLFWVVLLVYLVTLLGNSLIILLTQ...`

### Results:
- **True Location:** Cell membrane
- **Predicted Location:** Cell membrane
- **Confidence:** 100.00%

**Top 3 Predictions:**
1. Cell membrane (100.00%)
2. Vesicular (0.00%)
3. Cell projection (0.00%)

These examples demonstrate the model's high confidence in correctly predicting protein subcellular locations based on amino acid sequences.

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
pip install -r requirements.txt
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
python random_predictions.py --num_predictions <number of predictions you want to make>
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
