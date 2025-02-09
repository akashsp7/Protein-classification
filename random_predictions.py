import random
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import pandas as pd
import argparse

import config
from scripts import load_datasets, LocationLabelProcessor, ProteinLocalizationAnalyzer


def single_prediction(test_dataset, label_processor, tokenizer, model, df_single):
    
    # Getting random sample from test dataset
    random_idx = random.randint(0, len(test_dataset)-1)
    sample = test_dataset[random_idx]

    # Getting individual sequence from the test dataset (tokenized)
    sequence = sample['input_ids'] 
    
    true_label = sample['labels']
    true_label_name = label_processor.decode_labels(np.array([true_label]))[0]

    inputs = {
        'input_ids': torch.tensor(sequence).unsqueeze(0).to(device),
        'attention_mask': torch.tensor(sample['attention_mask']).unsqueeze(0).to(device)
    }

    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
        predicted_label = torch.argmax(predictions).item()
        predicted_label_name = label_processor.decode_labels(np.array([predicted_label]))[0]

    # Get probability scores
    probabilities = torch.nn.functional.softmax(predictions, dim=1)
    confidence = probabilities[0][predicted_label].item()

    # Find the Entry ID using the index from your original DataFrame
    original_idx = test_dataset[random_idx]['original_index'] if 'original_index' in test_dataset[random_idx] else random_idx
    entry_id = df_single['Entry'].iloc[original_idx]
    
    print(f"Entry: {entry_id}")
    sequence_text = tokenizer.decode(sequence, skip_special_tokens=True).replace(" ", "")
    print(f"Sequence: {sequence_text[:50]}...")
    print(f"\nTrue Label: {true_label_name}")
    print(f"Predicted Label: {predicted_label_name}")
    print(f"Confidence: {confidence:.2%}")

    # Get top 3 predictions
    top_3_probs, top_3_indices = torch.topk(probabilities[0], 3)
    print("\nTop 3 Predictions:")
    for prob, idx in zip(top_3_probs, top_3_indices):
        label_name = label_processor.decode_labels(np.array([idx.item()]))[0]
        print(f"{label_name}: {prob.item():.2%}")

# Initialize model and tokenizer
model_checkpoint = "facebook/esm2_t36_3B_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
device = 'cuda' if torch.cuda.is_available() else 'mps'
model = AutoModelForSequenceClassification.from_pretrained(config.model_path)
model = model.to(device)
model.eval()

# Load data
analyzer = ProteinLocalizationAnalyzer(pd.read_csv(config.csv_path))
df_single, _ = analyzer.create_dataframes()
label_processor = LocationLabelProcessor()
label_processor.fit(df_single['location_category'])

# Load the saved test dataset
_, test_dataset = load_datasets()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run predictions on test dataset')
    parser.add_argument('--num_predictions', '-n', type=int, default=10,
                      help='Number of predictions to make (default: 10)')
    
    args = parser.parse_args()
    print(f'\nStarting {args.num_predictions} Predictions...\n')
    for i in range(args.num_predictions):
        single_prediction(test_dataset, label_processor, tokenizer, model, df_single)
        print('-'*60)