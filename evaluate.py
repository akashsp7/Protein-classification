import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from tqdm import tqdm
from torch.utils.data import DataLoader
from scripts import load_datasets
import config

def calculate_top_n_accuracy(predictions, labels, n=5):
    """
    Calculate top-N accuracy from model predictions

    Args:
        predictions (torch.Tensor): Model predictions (batch_size x num_classes)
        labels (torch.Tensor): True labels
        n (int): Number of top predictions to consider

    Returns:
        float: Top-N accuracy score
    """
    # Get top N predictions
    _, top_n_preds = torch.topk(predictions, k=n, dim=1)

    # Convert labels to column vector
    labels = labels.view(-1, 1)

    # Check if true label is in top N predictions
    correct = (top_n_preds == labels).any(dim=1)

    return correct.float().mean().item()

def evaluate_model(model_path, test_dataset, tokenizer, ns=[1, 3, 5], batch_size=32, device=config.device):
    """
    Evaluate model with multiple top-N accuracy metrics

    Args:
        model_path (str): Path to saved model
        test_dataset (Dataset): HuggingFace dataset with test data
        tokenizer: HuggingFace tokenizer
        ns (list): List of N values for top-N accuracy
        batch_size (int): Batch size for evaluation
        device (str): Device to run evaluation on ('cuda' or 'cpu')

    Returns:
        dict: Dictionary with top-N accuracy scores
    """
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    # Create data collator with padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

    # Create dataloader with data_collator
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    all_predictions = []
    all_labels = []

    # Get predictions
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            # Get model predictions
            outputs = model(**inputs)
            logits = outputs.logits

            all_predictions.append(logits.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Calculate top-N accuracy for each N
    results = {}
    for n in ns:
        acc = calculate_top_n_accuracy(all_predictions, all_labels, n)
        results[f'top_{n}_accuracy'] = acc
        print(f'Top-{n} Accuracy: {acc:.4f}')

    return results

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
temp, test_dataset = load_datasets()
del temp

if __name__ == "__main__":
    # Run evaluation
    results = evaluate_model(
        model_path=config.model_path,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        ns=[1, 3, 5, 10],  # Calculate top-1, top-3, top-5, and top-10 accuracy
        batch_size=16,  # Smaller batch size to avoid OOM
        device='cuda' if torch.cuda.is_available() else 'mps'
    )