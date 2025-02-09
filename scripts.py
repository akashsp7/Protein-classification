'''
Our main two classes required for preparing data for training and evaluation.
'''

import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import torch

class ProteinLocalizationAnalyzer():

    def __init__(self, df):
        self.df = df
        self.categories = { # Broader categories of sub-cellular locations
            'Cell membrane': ['Cell membrane', 'Membrane', 'Plasma membrane', 'Cell surface',
                            'Peripheral membrane protein', 'Single-pass', 'Multi-pass'],
            'Golgi apparatus': ['Golgi', 'Trans-Golgi'],
            'Endoplasmic reticulum': ['Endoplasmic reticulum', 'ER', 'Sarcoplasmic reticulum'],
            'Nucleus': ['Nucleus', 'Nuclear', 'Chromosome', 'Nucleolus', 'Nucleolar',
                    'Chromatin', 'Ribonucleosome'],
            'Secreted': ['Secreted', 'Extracellular'],
            'Cytoplasm': ['Cytoplasm', 'Cytosol', 'Cytoplasmic', 'Perinuclear'],
            'Mitochondrion': ['Mitochondrion', 'Mitochondrial'],
            'Cell projection': ['Cell projection', 'Dendrite', 'Cilium', 'Axon', 'Flagellum',
                            'Lamellipodium', 'Filopodium'],
            'Vesicular': ['Lysosome', 'Endosome', 'Vesicle', 'Melanosome', 'Peroxisome',
                        'Autophagosome', 'Phagosome', 'Cytolytic granule',
                        'Preautophagosomal', 'Microsome'],
            'Synapse': ['Synapse', 'Synaptosome', 'Postsynaptic', 'Presynaptic'],
            'Cell junction': ['Cell junction', 'Tight junction', 'Gap junction',
                            'Focal adhesion', 'Desmosome'],
            'Lipid-anchor': ['Lipid-anchor', 'GPI-anchor', 'Lipid droplet']
        }
        self.special_cases = {         # Words/phrases that indicate a location should be categorized under a specific category
            'Sarcoplasmic reticulum': 'Endoplasmic reticulum',
            'Associates with chromatin': 'Nucleus',
            'Component of ribonucleosomes': 'Nucleus',
            'Midbody': 'Cell projection',  # Could be debatable, but midbody is a cellular projection during division
        }

        # Priority ordering for when multiple categories are found
        self.priority_order = [
            'Lipid-anchor',  # Most specific
            'Synapse',
            'Cell junction',
            'Cell projection',
            'Vesicular',
            'Golgi apparatus',
            'Endoplasmic reticulum',
            'Mitochondrion',
            'Nucleus',
            'Secreted',
            'Cell membrane',
            'Cytoplasm'      # Most general
        ]

    def extract_locations(self, location_str):
        """
        Extract core subcellular locations from the location string.

        Args:
            location_str (str): Raw location string from UniProt

        Returns:
            list: List of unique core locations
        """
        # Remove the prefix (Present in every single row)
        location_str = location_str.replace('SUBCELLULAR LOCATION:', '')

        # Split by semicolon to separate different parts
        parts = location_str.split(';')

        locations = []
        for part in parts:
            # Skip empty parts and notes
            if not part.strip() or part.strip().startswith('Note='):
                continue

            # Remove evidence codes {ECO:...}
            part = re.sub(r'\{ECO:[^}]+\}', '', part)

            # Remove isoform specifics
            part = re.sub(r'\[Isoform [^\]]+\]:', '', part)

            # Remove protein type details
            part = re.sub(r'(Single|Multi)-pass.*protein', '', part)

            # Split by period to handle multiple locations in one part
            subparts = part.split('.')

            for subpart in subparts:
                # Clean and get the core location
                loc = subpart.strip()
                if loc:
                    locations.append(loc)

        # Clean up locations and remove duplicates
        clean_locations = []
        for loc in locations:
            # Remove leading/trailing whitespace and periods
            loc = loc.strip(' .')
            if loc and loc not in clean_locations:
                clean_locations.append(loc)

        return clean_locations

    def analyze_locations(self):
        """
        Analyze subcellular locations in the dataframe.

        Returns:
            dict: Dictionary with location counts
        """
        all_locations = []

        for loc_str in self.df['Subcellular location [CC]']:
            locations = self.extract_locations(loc_str)
            all_locations.extend(locations)

        # Count unique locations
        location_counts = {}
        for loc in all_locations:
            location_counts[loc] = location_counts.get(loc, 0) + 1

        # Sort by frequency
        sorted_counts = dict(sorted(location_counts.items(),
                                key=lambda x: x[1],
                                reverse=True))

        return sorted_counts

    def categorize_locations(self, location_counts):
        """
        Categorize locations into major groups and count their frequencies.
        Also identifies uncategorized locations.
        """

        # Initialize counts
        category_counts = {cat: 0 for cat in self.categories}
        uncategorized = {}

        # Process each location and its count
        for loc, count in location_counts.items():
            categorized = False

            # Check special cases
            for special_case, category in self.special_cases.items():
                if special_case.lower() in loc.lower():
                    category_counts[category] += count
                    categorized = True
                    break

            if not categorized:
                # Check each category
                for cat, keywords in self.categories.items():
                    # If any keyword from the category is in the location
                    if any(keyword.lower() in loc.lower() for keyword in keywords):
                        category_counts[cat] += count
                        categorized = True
                        break

            # If location wasn't categorized, add taht to uncategorized
            if not categorized:
                uncategorized[loc] = count

        # Sort both dictionaries by count
        category_counts = dict(sorted(category_counts.items(),
                                    key=lambda x: x[1],
                                    reverse=True))
        uncategorized = dict(sorted(uncategorized.items(),
                                key=lambda x: x[1],
                                reverse=True))

        return category_counts, uncategorized

    def print_categorization_summary(self, category_counts, uncategorized):
        """
        Print a summary of the categorization results with percentages
        """
        print("=== ANALYSIS OF DATASET")
        print("=== MAJOR CATEGORIES ===")
        total_categorized = sum(category_counts.values())
        for cat, count in category_counts.items():
            percentage = (count/total_categorized)*100
            print(f"{cat}: {count} ({percentage:.1f}%)")

        print("\n=== UNCATEGORIZED LOCATIONS ===")
        total_uncategorized = sum(uncategorized.values())
        print(f"Top 20 uncategorized locations (out of {len(uncategorized)}):")
        for loc, count in list(uncategorized.items())[:20]:
            print(f"{loc}: {count}")

        print("\n=== SUMMARY ===")
        total = total_categorized + total_uncategorized
        print(f"Total categorized: {total_categorized} ({(total_categorized/total)*100:.1f}%)")
        print(f"Total uncategorized: {total_uncategorized} ({(total_uncategorized/total)*100:.1f}%)")

    def assign_single_location_category(self, location_str):
        """
        Assign a single category to a location string based on priority order.

        Args:
            location_str (str): Raw location string from UniProt

        Returns:
            str: Single category name or 'Uncategorized'
        """
        # Extract all locations using our existing function
        locations = self.extract_locations(location_str)

        found_categories = set()

        # Process each extracted location
        for loc in locations:
            # Check special cases
            for special_case, category in self.special_cases.items():
                if special_case.lower() in loc.lower():
                    found_categories.add(category)
                    break

            # Check each category's keywords
            for category, keywords in self.categories.items():
                if any(keyword.lower() in loc.lower() for keyword in keywords):
                    found_categories.add(category)

        if not found_categories:
            return 'Uncategorized'

        # Return the highest priority category found
        for category in self.priority_order:
            if category in found_categories:
                return category

        return list(found_categories)[0]  # Fallback to first found category

    def assign_multi_location_categories(self, location_str):
        """
        Assign multiple categories to a location string.

        Args:
            location_str (str): Raw location string from UniProt

        Returns:
            dict: Dictionary with category names as keys and 1/0 as values
        """

        # Initialize results dictionary with all categories set to 0
        results = {category: 0 for category in self.categories.keys()}

        # Extract locations
        locations = self.extract_locations(location_str)

        # Process each location
        for loc in locations:
            # Check special cases
            for special_case, category in self.special_cases.items():
                if special_case.lower() in loc.lower():
                    results[category] = 1
                    break

            # Check each category
            for category, keywords in self.categories.items():
                if any(keyword.lower() in loc.lower() for keyword in keywords):
                    results[category] = 1

        return results

    def add_location_categories(self, multilabel=False):
        """
        Add location categories to the DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame with 'Subcellular location [CC]' column
            multilabel (bool): If True, add multiple binary columns for each category
                            If False, add single category column

        Returns:
            pandas.DataFrame: DataFrame with new location category column(s)
        """
        df_with_categories = self.df.copy()

        if multilabel:
            # Get multilabel categories for each row
            multi_categories = df_with_categories['Subcellular location [CC]'].apply(self.assign_multi_location_categories)

            # Convert the series of dictionaries to multiple columns
            for category in multi_categories.iloc[0].keys():
                df_with_categories[f'location_{category.lower().replace(" ", "_")}'] = multi_categories.apply(lambda x: x[category])
        else:
            # Single label classification
            df_with_categories['location_category'] = df_with_categories['Subcellular location [CC]'].apply(self.assign_single_location_category)

        return df_with_categories

    def create_dataframes(self):
        """
        Create single and multi-label dataframes.

        Returns:
            pandas.DataFrame: Single label Dataframe,
            pandas.DataFrame: Multi-label Dataframe.
        """
        sorted_counts = self.analyze_locations()
        category_counts, uncategorized = self.categorize_locations(sorted_counts)
        self.print_categorization_summary(category_counts, uncategorized)
        # For single-label classification:
        df_single = self.add_location_categories(multilabel=False)

        # For multi-label classification:
        df_multi = self.add_location_categories(multilabel=True)
        return df_single, df_multi

class LocationLabelProcessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.num_classes = None
        self.class_to_idx = None
        self.idx_to_class = None

    def fit(self, labels):
        """
        Fit the label encoder to the location categories.

        Args:
            labels (pandas.Series): Location category labels
        """
        # Removing 'Uncategorized' labels before fitting
        clean_labels = labels[labels != 'Uncategorized']

        # Fit 
        self.label_encoder.fit(clean_labels)

        # Storing number of classes to later initialize model head
        self.num_classes = len(self.label_encoder.classes_)

        # Creating mappings
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.label_encoder.classes_)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        return self

    def encode_labels(self, labels):
        """
        Convert location labels to numeric indices.

        Args:
            labels (pandas.Series): Location category labels

        Returns:
            torch.tensor: Tensor of label indices
        """
        # Filter out uncategorized samples
        mask = labels != 'Uncategorized'
        valid_labels = labels[mask]

        # Convert to numeric indices
        label_indices = self.label_encoder.transform(valid_labels)

        # Convert to tensor
        return torch.tensor(label_indices, dtype=torch.long), mask

    def decode_labels(self, indices):
        """
        Convert numeric indices back to location categories.

        Args:
            indices (torch.tensor or numpy.array): Numeric label indices

        Returns:
            numpy.array: Array of location category labels
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()
        return self.label_encoder.inverse_transform(indices)

    def get_class_weights(self, labels):
        """
        Calculate class weights for handling imbalanced data.

        Args:
            labels (pandas.Series): Location category labels

        Returns:
            torch.tensor: Tensor of class weights
        """
        # Filter out uncategorized samples
        valid_labels = labels[labels != 'Uncategorized']

        # Get class counts
        class_counts = valid_labels.value_counts()

        # Calculate weights (inverse of frequency)
        total_samples = len(valid_labels)
        weights = torch.zeros(self.num_classes)
        for cls, count in class_counts.items():
            idx = self.class_to_idx[cls]
            weights[idx] = total_samples / (self.num_classes * count)

        return weights
    
import pickle
from pathlib import Path

def save_datasets(train_dataset, test_dataset, save_dir='datasets'):
    """
    Save train and test datasets to pickle files
    
    Args:
        train_dataset: HuggingFace dataset for training
        test_dataset: HuggingFace dataset for testing 
        save_dir: Directory to save the pickle files (will be created if doesn't exist)
    """
    # Create directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Save train dataset
    with open(save_path / 'train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    
    # Save test dataset    
    with open(save_path / 'test_dataset.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)
    print(f'\nBoth datasets saved to {save_path}')
        
def load_datasets(load_dir='datasets'):
    """
    Load train and test datasets from pickle files
    
    Args:
        load_dir: Directory containing the pickle files
        
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    load_path = Path(load_dir)
    
    # Load train dataset
    with open(load_path / 'train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    
    # Load test dataset
    with open(load_path / 'test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
        
    print(f'\nBoth datasets loaded from {load_path}')
        
    return train_dataset, test_dataset