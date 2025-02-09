import os

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset

from scripts import ProteinLocalizationAnalyzer, LocationLabelProcessor, save_datasets
import config


analyzer = ProteinLocalizationAnalyzer(pd.read_csv(config.csv_path))
label_processor = LocationLabelProcessor()

if __name__ == '__main__':
    df_single, temp = analyzer.create_dataframes()
    del temp    
    
    label_processor.fit(df_single['location_category'])
    labels, valid_mask = label_processor.encode_labels(df_single['location_category'])
    sequences = df_single['Sequence'][valid_mask]
    class_weights = label_processor.get_class_weights(df_single['location_category'])
    
    print(f"\nNumber of classes: {label_processor.num_classes}")
    print(f"Class mapping: {label_processor.class_to_idx}")
    
    # No validation set since I did not want to train the model twice. 
    train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.2, shuffle=True)

    model_checkpoint = "facebook/esm2_t36_3B_UR50D" # Requires A100 (Colab pro version). Haven't tried using LORA and running on T4.
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # Tokenize
    train_tokenized = tokenizer(train_sequences.tolist()) 
    test_tokenized = tokenizer(test_sequences.tolist())

    # Convert to dataset object
    train_dataset = Dataset.from_dict(train_tokenized)
    test_dataset = Dataset.from_dict(test_tokenized)

    # Add labels to dataset
    train_dataset = train_dataset.add_column("labels", train_labels.tolist())
    test_dataset = test_dataset.add_column("labels", test_labels.tolist())
    save_datasets(train_dataset, test_dataset)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=label_processor.num_classes)
    
    # Mainly for avoiding the out of memory error despite using A100. Definitely didn't need on 150M model.
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    from transformers import get_linear_schedule_with_warmup
    model_name = model_checkpoint.split("/")[-1]

    batch_size = 4  
    gradient_accumulation_steps = 4  # So basically 16. Other combinations making it 32 did not work.
    num_epochs = 5

    # Calculate training steps
    steps_per_epoch = len(train_dataset) // (batch_size * gradient_accumulation_steps)
    total_training_steps = steps_per_epoch * num_epochs
    warmup_steps = total_training_steps // 10

    # Enable gradient checkpointing (Basically not storing every single activation in forward pass trading off speed for memory efficiency)
    model.gradient_checkpointing_enable()

    args = TrainingArguments(
        f"{model_name}-finetuned",
        learning_rate=1e-4, 
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,  
        gradient_accumulation_steps=gradient_accumulation_steps, # Simulating batch size 16 (4*4) 
        num_train_epochs=num_epochs,
        warmup_steps=warmup_steps,
        weight_decay=0.01, # Good practice, better generalization
        lr_scheduler_type='linear', # linear decrease of our initial learning rate
        evaluation_strategy="epoch", # Dint use steps despite calcs above since this saved time
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True, # Precision 16
        report_to='none', # Can use wandb or tensorboard but I wasn't going to analyze much, so useless
        gradient_checkpointing=True,
        optim="adafactor",  # Adam taking too much memory, this uses less memory by not storing full optimizer states
        max_grad_norm=1.0, # Clipping gradients
        save_total_limit=2,  
        dataloader_num_workers=0, # Slower dataloading but less memory use
        remove_unused_columns=True,
    )

    # Add empty cache calls
    torch.cuda.empty_cache()
    
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=2,
        early_stopping_threshold=0.01
    )

    # Create trainer with callbacks
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset, # This was a mistake
        tokenizer=tokenizer,
        callbacks=[early_stopping]
    )
    
    trainer.train()
    model.save_pretrained(config.model_path)