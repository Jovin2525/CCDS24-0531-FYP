import os
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from Bio import SeqIO
from gensim.models import word2vec
import warnings
import joblib
import pickle
import sys
import json
from deep_network import Transformer_PHV
from metrics import cofusion_matrix, sensitivity, specificity, auc, mcc, accuracy, precision, recall, f1, cutoff, AUPRC

warnings.simplefilter('ignore')

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
amino_acids_vector = np.eye(20)
normal_amino_acids = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]

def file_input_csv(filename, index_col=None):
    """Load CSV file into pandas DataFrame."""
    data = pd.read_csv(filename, index_col=index_col)
    return data

def load_joblib(filename):
    """Load joblib file."""
    with open(filename, "rb") as f:
        return joblib.load(f)

def save_joblib(filename, data):
    """Save data to joblib file."""
    with open(filename, "wb") as f:
        joblib.dump(data, f, compress=3)

def output_csv(filename, data):
    """Save DataFrame to CSV."""
    data.to_csv(filename, index=False)

def transform_seq_scale(seq, stride):
    """Transform sequence length based on stride."""
    return int(seq/stride) + (1 if seq % stride != 0 else 0)

def binary_embed(seq):
    """Create binary embedding for amino acid sequence."""
    return np.identity(len(normal_amino_acids))[[normal_amino_acids.index(s) for s in seq]]

def w2v_encoding(seq, model, window, step, k_mer):
    """Encode sequence using word2vec model."""
    vecs = []
    for i in range(0, len(seq), step):
        pep = seq[i:i + window]
        if len(pep) >= k_mer:
            vec = np.mean([model.wv[pep[j:j + k_mer]] for j in range(len(pep) - k_mer + 1)], axis=0)
            vecs.append(vec)
    return vecs

def create_mat_dict_w2v(seqs, enc_model, k_mer, filepath="encoded_sequences.pkl"):
    """Create matrix dictionary for sequences using word2vec model."""
    if os.path.exists(filepath):
        print(f"Loading encoded sequences from {filepath}")
        with open(filepath, "rb") as f:
            return pickle.load(f)
    
    print("Encoding sequences...")
    seqs = list(set(seqs))
    seq2mat_dict = {}
    
    for i in range(len(seqs)):
        seq2mat_dict[seqs[i]] = torch.tensor([
            enc_model.wv[seqs[i][j:j + k_mer]] 
            for j in range(len(seqs[i]) - k_mer + 1)
        ])
        
        if (i + 1) % 1000 == 0:
            print(f"Encoded {i + 1}/{len(seqs)} sequences")

    print(f"Saving encoded sequences to {filepath}")
    with open(filepath, "wb") as f:
        pickle.dump(seq2mat_dict, f)

    return seq2mat_dict

def encoding_antibody_chains(heavy_chain_seq, light_chain_seq, enc_model, enc_seq_max, window, stride, k_mer):
    """Encode antibody chains for model input."""
    # Get sequences from dictionary
    heavy_chain_mat = enc_model[heavy_chain_seq].to(device)
    light_chain_mat = enc_model[light_chain_seq].to(device)
    
    # Calculate sequence lengths
    mat_len_heavy, mat_len_light = len(heavy_chain_mat), len(light_chain_mat)
    w2v_seq_max = enc_seq_max - k_mer + 1
    
    # Pad sequences
    heavy_chain_mat = torch.nn.functional.pad(
        heavy_chain_mat, 
        (0, 0, 0, w2v_seq_max - mat_len_heavy)
    ).float()
    
    light_chain_mat = torch.nn.functional.pad(
        light_chain_mat, 
        (0, 0, 0, w2v_seq_max - mat_len_light)
    ).float()
    
    # Calculate convolution lengths
    mat_conv_len_heavy = max(int((mat_len_heavy - window) / stride) + 1, 1)
    mat_conv_len_light = max(int((mat_len_light - window) / stride) + 1, 1)
    max_conv_len = int((w2v_seq_max - window) / stride) + 1

    # Create attention masks
    w2v_attn_mask_heavy = torch.cat((
        torch.zeros((mat_conv_len_heavy, max_conv_len), device=device).long(),
        torch.ones((max_conv_len - mat_conv_len_heavy, max_conv_len), device=device).long()
    )).transpose(-1, -2).bool()
    
    w2v_attn_mask_light = torch.cat((
        torch.zeros((mat_conv_len_light, max_conv_len), device=device).long(),
        torch.ones((max_conv_len - mat_conv_len_light, max_conv_len), device=device).long()
    )).transpose(-1, -2).bool()
    
    # Move tensors back to CPU
    heavy_chain_mat = heavy_chain_mat.cpu()
    light_chain_mat = light_chain_mat.cpu()
    w2v_attn_mask_heavy = w2v_attn_mask_heavy.cpu()
    w2v_attn_mask_light = w2v_attn_mask_light.cpu()
    
    return heavy_chain_mat, light_chain_mat, w2v_attn_mask_heavy, w2v_attn_mask_light

class antibody_data_sets(data.Dataset):
    """Dataset class for antibody data."""
    def __init__(self, data_sets, enc_model, enc_seq_max=9000, window=20, stride=10, k_mer=1):
        super().__init__()
        self.heavy_chain = data_sets["heavy_chain"].values.tolist()
        self.light_chain = data_sets["light_chain"].values.tolist()
        self.enc_model = enc_model
        self.enc_seq_max = enc_seq_max
        self.window = window
        self.stride = stride
        self.k_mer = k_mer

    def __len__(self):
        return len(self.heavy_chain)

    def __getitem__(self, idx):
        heavy_chain_mat, light_chain_mat, attn_mask_heavy, attn_mask_light = encoding_antibody_chains(
            self.heavy_chain[idx],
            self.light_chain[idx],
            self.enc_model,
            self.enc_seq_max,
            self.window,
            self.stride,
            self.k_mer
        )
        return (
            idx,
            self.heavy_chain[idx],
            self.light_chain[idx],
            heavy_chain_mat.to(device),
            light_chain_mat.to(device),
            attn_mask_heavy.to(device),
            attn_mask_light.to(device)
        )

class DeepNet():
    """Deep neural network model for antibody prediction."""
    def __init__(self, out_path, enc_dict, deep_path, model_params, prediction_params, encoding_params, vec_ind):
        self.out_path = out_path
        self.enc_model = enc_dict
        self.deep_path = deep_path
        self.vec_ind = vec_ind
        self.model_params = model_params
        self.batch_size = prediction_params["batch_size"]
        self.enc_seq_max = encoding_params["enc_seq_max"]
        self.thresh = prediction_params["thresh"]
        self.k_mer = encoding_params["k_mer"]
        
    def model_training(self, data_sets):
        """Run predictions on the dataset and calculate metrics if ground truth labels are available."""
        os.makedirs(self.out_path, exist_ok=True)
        os.makedirs(self.out_path + "/logs", exist_ok=True)
        
        # Save the original stdout
        original_stdout = sys.stdout
        
        log_file = open(self.out_path + "/logs/prediction.log", "w")
        sys.stdout = log_file

        try:
            # Initialize dataset and dataloader
            print("Creating dataset...", flush=True)
            data_all = antibody_data_sets(
                data_sets, 
                enc_model=self.enc_model, 
                enc_seq_max=self.enc_seq_max, 
                window=self.model_params["kernel_size"], 
                stride=self.model_params["stride"], 
                k_mer=self.k_mer
            )
            loader = DataLoader(dataset=data_all, batch_size=self.batch_size)
        
            # Initialize model on GPU
            print("Initializing model...", flush=True)
            self.model = Transformer_PHV(
                filter_num=self.model_params["filter_num"],
                kernel_size_w2v=self.model_params["kernel_size"],
                stride_w2v=self.model_params["stride"],
                n_heads=self.model_params["n_heads"],
                d_dim=self.model_params["d_dim"],
                feature=self.model_params["feature"],
                pooling_dropout=self.model_params["pooling_dropout"],
                linear_dropout=self.model_params["linear_dropout"]
            ).to(device)

            # Load model state - handle checkpoint format
            print(f"Loading model from {self.deep_path}...", flush=True)
            try:
                checkpoint = torch.load(self.deep_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print("Loaded model from checkpoint format", flush=True)
                else:
                    self.model.load_state_dict(checkpoint)
                    print("Loaded model from direct state dict format", flush=True)
            except Exception as e:
                print(f"Error loading model: {str(e)}", flush=True)
                raise e
        
            probs_all = []
            
            if self.vec_ind:
                h_out_1_list, h_out_2_list = [], []
                out_1_list, out_2_list, out_list = [], [], []

            print("Running predictions...", flush=True)
            self.model.eval()
            with torch.no_grad():
                for i, (idx, heavy_chain_seq, light_chain_seq, heavy_chain_mat, light_chain_mat, attn_mask_heavy, attn_mask_light) in enumerate(loader):
                    if i % 10 == 0:
                        print(f"Processing batch {i}/{len(loader)}", flush=True)
                        
                    try:
                        # Forward pass
                        probs = self.model(heavy_chain_mat, light_chain_mat, attn_mask_heavy, attn_mask_light)
                        
                        # Store results
                        batch_probs = probs.cpu().detach().squeeze(1).numpy().flatten().tolist()
                        probs_all.extend(list(zip(idx.cpu().numpy(), heavy_chain_seq, light_chain_seq, batch_probs)))
                        
                        if self.vec_ind:
                            # Store intermediate outputs
                            h_out_1_list.extend(self.model.h_out_1.cpu().detach().numpy())
                            h_out_2_list.extend(self.model.h_out_2.cpu().detach().numpy())
                            out_1_list.extend(self.model.out_1.cpu().detach().numpy())
                            out_2_list.extend(self.model.out_2.cpu().detach().numpy())
                            out_list.extend(self.model.out.cpu().detach().numpy())

                    except Exception as e:
                        print(f"Error processing batch {i}: {str(e)}")
                        continue
                    finally:
                        # Clear GPU memory
                        del heavy_chain_mat, light_chain_mat, attn_mask_heavy, attn_mask_light
                        if 'probs' in locals():
                            del probs
                        torch.cuda.empty_cache()

            # Convert results to DataFrame and save
            print("Saving results...", flush=True)
            probs_all = pd.DataFrame(
                probs_all, 
                columns=["idx", "heavy_chain_seq", "light_chain_seq", "scores"]
            )
            output_csv(self.out_path + "/probs.csv", probs_all)
                
            if self.vec_ind:
                # Save intermediate outputs if requested
                print("Saving feature vectors...", flush=True)
                save_joblib(self.out_path + "/after_cnn_heavy.joblib", np.array(h_out_1_list))
                save_joblib(self.out_path + "/after_cnn_light.joblib", np.array(h_out_2_list))
                save_joblib(self.out_path + "/feature_vec_heavy.joblib", np.array(out_1_list))
                save_joblib(self.out_path + "/feature_vec_light.joblib", np.array(out_2_list))
                save_joblib(self.out_path + "/concatenated_feature_vec.joblib", np.array(out_list))
            
            # Calculate metrics if ground truth labels are available
            if "paired" in data_sets.columns:
                print("Calculating metrics...", flush=True)
                true_labels = data_sets["paired"].values.tolist()
                predicted_probs = probs_all["scores"].values.tolist()
                
                # Calculate and save metrics
                metrics_results = {}
                print("Threshold:", self.thresh, flush=True)
                
                # Add confusion matrix values
                tn, fp, fn, tp = cofusion_matrix(true_labels, predicted_probs, thresh=self.thresh)
                metrics_results["true_negative"] = tn
                metrics_results["false_positive"] = fp
                metrics_results["false_negative"] = fn
                metrics_results["true_positive"] = tp
                
                # Calculate standard metrics
                metrics_dict = {
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mcc": mcc,
                    "auc": auc,
                    "AUPRC": AUPRC
                }
                
                for key, metric_func in metrics_dict.items():
                    if key not in ["auc", "AUPRC"]:
                        value = metric_func(true_labels, predicted_probs, thresh=self.thresh)
                    else:
                        value = metric_func(true_labels, predicted_probs)
                    metrics_results[key] = value
                    print(f"{key}: {value}", flush=True)
                
                # Find optimal thresholds
                threshold_auc, threshold_prc = cutoff(true_labels, predicted_probs)
                metrics_results["optimal_threshold_auc"] = threshold_auc
                metrics_results["optimal_threshold_prc"] = threshold_prc
                print(f"Optimal threshold (AUC): {threshold_auc}", flush=True)
                print(f"Optimal threshold (PRC): {threshold_prc}", flush=True)
                
                # Save metrics as CSV
                metrics_df = pd.DataFrame([metrics_results])
                output_csv(self.out_path + "/metrics.csv", metrics_df)
                
                # Convert NumPy values to Python native types for JSON serialization
                for key in metrics_results:
                    if isinstance(metrics_results[key], np.number):
                        metrics_results[key] = metrics_results[key].item()
                
                # Save metrics as JSON for easier reading
                with open(self.out_path + "/metrics.json", 'w') as f:
                    json.dump(metrics_results, f, indent=4)
            
            return True
            
        finally:
            sys.stdout = original_stdout
            log_file.close()

def pred_main(in_path, out_path, w2v_model_path, deep_model_path, vec_ind, thresh, batch_size, k_mer, seq_max):
    """Main prediction function with metrics calculation."""
    # Model parameters
    model_params = {
        "filter_num": 128,
        "kernel_size": 20,
        "stride": 10,
        "n_heads": 4,
        "d_dim": 32,
        "feature": 128,
        "pooling_dropout": 0.5,
        "linear_dropout": 0.3
    }
    prediction_params = {
        "batch_size": batch_size,
        "thresh": thresh
    }
    encoding_params = {
        "enc_seq_max": seq_max,
        "k_mer": k_mer
    }

    print("Loading datasets", flush=True)
    data = file_input_csv(in_path)

    print("Loading word2vec model", flush=True)
    w2v_model = word2vec.Word2Vec.load(w2v_model_path)

    print("Encoding amino acid sequences", flush=True)
    embedding_path = "./encoded_sequences.pkl"
    enc_dict = create_mat_dict_w2v(
        data["heavy_chain"].values.tolist() + data["light_chain"].values.tolist(),
        w2v_model,
        encoding_params["k_mer"],
        embedding_path
    )

    print("Start prediction", flush=True)
    net = DeepNet(
        out_path,
        enc_dict,
        deep_model_path,
        model_params,
        prediction_params,
        encoding_params,
        vec_ind
    )
    res = net.model_training(data)

    print("Finish processing", flush=True)
    return res
