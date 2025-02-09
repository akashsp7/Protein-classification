'''
Change various paths by editing this script
'''

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

if device == 'cuda':
    csv_path = '/content/drive/MyDrive/Protein_folding/protein_folding.csv'
    model_path = '/content/drive/MyDrive/Protein_folding'
else:
    csv_path = 'protein_folding.csv'
    model_path = 'Model'
    
    