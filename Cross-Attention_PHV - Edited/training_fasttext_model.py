import os
import pandas as pd
from Bio import SeqIO
from gensim.models import FastText 
import logging

normal_amino_acids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", 
                      "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

def file_input_csv(filename, index_col=None):
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(filename, index_col=index_col)

def import_dataset(filepath):
    """Import a dataset and extract the 'AA_Sequence' column."""
    dataset = pd.read_csv(filepath, header=None)
    sequences = dataset[0].tolist()
    return sequences

def mer_vec(seq, k_mer=4):
    """Convert sequences into overlapping k-mers."""
    return [[seq[i][j:j + k_mer] for j in range(len(seq[i]) - k_mer + 1)] for i in range(len(seq))]

def training_fasttext(data_path, out_path, k_mer=4, vector_size=128, window_size=5, iteration=100):
    """Train a FastText model on k-mers of sequences."""
    # Import sequences and generate k-mers
    seq_list = import_dataset(data_path)
    k_mers_list = mer_vec(seq_list, k_mer=k_mer)

    # Configure logging to monitor the training process
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Train the FastText model
    model = FastText(sentences=k_mers_list, 
                     vector_size=vector_size, 
                     window=window_size, 
                     min_count=1,  # Include all k-mers
                     sg=1,  # Use skip-gram
                     epochs=iteration)

    # Ensure the output directory exists and save the trained model
    os.makedirs(out_path, exist_ok=True)
    model.save(os.path.join(out_path, "AA_model_fasttext.pt"))

