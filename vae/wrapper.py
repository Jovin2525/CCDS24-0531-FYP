import os
import numpy as np
import torch
from datetime import datetime
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align.Applications import MuscleCommandline
import random
import subprocess
from collections import defaultdict
from models.vae import VariationalAutoencoder
from data.helper import DataHelper
from utils.train import train
import torch.nn.functional as F

# Set global random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class SequenceLoader:
    def __init__(self):
        self.sequences_df = None
    
    def load_sequences(self, train_path: str) -> Tuple[List[str], List[str]]:
        """Load sequences from CSV with AA_Sequence, heavy_chain, and light_chain columns"""
        try:
            df_train = pd.read_csv(train_path)
            
            required_cols = ["AA_Sequence", "heavy_chain", "light_chain"]
            if not all(col in df_train.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")
            
            heavy_sequences = df_train["heavy_chain"].dropna().tolist()
            light_sequences = df_train["light_chain"].dropna().tolist()
            
            print(f"Loaded {len(heavy_sequences)} heavy chains and {len(light_sequences)} light chains")
            
            return heavy_sequences, light_sequences
            
        except Exception as e:
            raise Exception(f"Error loading CSV file: {str(e)}")

class DeepSequenceWrapper:
    def __init__(self, alignment_file: str, working_dir: str):
        """Initialize DeepSequence wrapper"""
        # Set random seeds
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize data helper
        self.data = DataHelper(
            alignment_file=alignment_file,
            working_dir=working_dir,
            calc_weights=True,
            theta=0.2,
            batch_size=1000  # Large batch size for efficiency
        )
        
        # Create necessary directories
        os.makedirs(os.path.join(working_dir, "params"), exist_ok=True)
        os.makedirs(os.path.join(working_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(working_dir, "embeddings"), exist_ok=True)
        
        # Model parameters from DeepSequence defaults
        self.model_params = {
            'encoder_architecture': [100, 100],
            'decoder_architecture': [100, 500],
            'n_latent': 30,
            'n_patterns': 1,
            'batch_size': 100,
            'encode_nonlinearity_type': "relu",
            'decode_nonlinearity_type': "relu",
            'final_decode_nonlinearity': "sigmoid",
            'sparsity': False,
            'convolve_encoder': False,
            'convolve_patterns': False
            # Removed device from model_params
        }
        
        self.model = None
        self.working_dir = working_dir

    def train_model(
        self,
        num_updates: int = 200000,
        patience: int = 10000,
        min_delta: float = 0.01,
        verbose: bool = True,
        save_progress: bool = True,
        save_parameters: bool = False,
        job_string: str = "",
        embeddings: Union[bool, str, int] = False,
        update_offset: int = 0,
        print_neff: bool = True,
        print_iter: int = 100
    ):
        """Train the VAE model"""
        print("\nInitializing model...")
        
        # Create model and move to device after initialization
        self.model = VariationalAutoencoder(
            data=self.data,
            working_dir=self.working_dir,
            **self.model_params
        ).to(self.device)

        # Call train function with parameters
        train(
            data=self.data,
            model=self.model,
            save_progress=save_progress,
            save_parameters=save_parameters,
            num_updates=num_updates,
            verbose=verbose,
            job_string=job_string,
            embeddings=embeddings,
            update_offset=update_offset,
            print_neff=print_neff,
            print_iter=print_iter
        )
        
    def generate_negative_samples(
        self, 
        n_samples: int = 100, 
        temp: float = 0.8, 
        min_dist: float = 1.5
    ) -> List[str]:
        """Generate negative samples using trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Sample from latent space
        z = torch.randn(n_samples * 2, self.model_params['n_latent'], device=self.device)
        
        # Get training data embeddings
        x_train_tensor = torch.tensor(self.data.x_train, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mu, _ = self.model.encode(x_train_tensor)
            train_z = mu
        
        # Filter samples based on distance
        valid_z = []
        
        # Convert to numpy for distance calculations
        z_cpu = z.cpu().numpy()
        train_z_cpu = train_z.cpu().numpy()
        
        for zi in z_cpu:
            distances = np.linalg.norm(train_z_cpu - zi, axis=1)
            if np.min(distances) >= min_dist:
                valid_z.append(zi)
            if len(valid_z) >= n_samples:
                break
        
        if len(valid_z) < n_samples:
            print(f"Warning: Only generated {len(valid_z)} valid samples")
        
        # Generate sequences from latent vectors
        sequences = []
        
        # Convert valid_z to tensor properly
        valid_z_array = np.array(valid_z[:n_samples])
        valid_z = torch.tensor(valid_z_array, dtype=torch.float32, device=self.device)
        
        # Generate in batches
        batch_size = min(1000, len(valid_z))
        with torch.no_grad():
            for i in range(0, len(valid_z), batch_size):
                batch_z = valid_z[i:i+batch_size]
                
                # Use decode instead of forward
                h = self.model.decoder(batch_z)
                if self.model.convolve_patterns:
                    # Reshape for pattern convolution
                    W = self.model.W_out.weight.reshape(
                        self.model.final_output_size,
                        self.model.seq_len,
                        self.model.conv_decoder_size
                    )
                    if self.model.sparsity:
                        if self.model.sparsity == "logit":
                            scale = torch.sigmoid(self.model.W_out_scale)
                        else:
                            scale = torch.exp(self.model.W_out_scale)
                        scale = scale.repeat(self.model.n_patterns, 1, 1)
                        W = W * scale
                    W = self.model.conv_decode(W)
                else:
                    W = self.model.W_out.weight.reshape(
                        self.model.final_output_size,
                        self.model.seq_len,
                        self.model.alphabet_size
                    )
                    if self.model.sparsity:
                        if self.model.sparsity == "logit":
                            scale = torch.sigmoid(self.model.W_out_scale)
                        else:
                            scale = torch.exp(self.model.W_out_scale)
                        scale = scale.repeat(self.model.n_patterns, 1, 1)
                        W = W * scale
                
                # Reshape for matrix multiplication
                W = W.reshape(self.model.final_output_size, -1)
                h_flat = torch.matmul(h, W)
                
                if self.model.output_bias:
                    h_flat = h_flat + self.model.b_out
                    
                if self.model.final_pwm_scale:
                    h_flat = h_flat * F.softplus(self.model.final_pwm_scale_param)
                
                # Reshape and apply temperature-scaled softmax
                logits = h_flat.reshape(-1, self.model.seq_len, self.model.alphabet_size)
                logits = logits / temp
                probs = F.softmax(logits, dim=-1)
                
                # Convert probabilities to sequences
                probs_np = probs.cpu().numpy()
                for seq_probs in probs_np:
                    sequence = ""
                    for pos_probs in seq_probs:
                        aa_idx = np.random.choice(len(pos_probs), p=pos_probs)
                        sequence += self.data.alphabet[aa_idx]
                    sequences.append(sequence)
        
        return sequences

class PairedNegativeGenerator:
    def __init__(self, output_dir: str = "./antibody_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.sequence_loader = SequenceLoader()
        random.seed(42)  # Set random seed for reproducibility
    
    def process_sequences_to_msa(self, sequences: List[str], chain_type: str) -> str:
        """Convert sequences to MSA using MUSCLE v5 with caching"""
        chain_dir = os.path.join(self.output_dir, chain_type)
        os.makedirs(chain_dir, exist_ok=True)
        
        # Define paths for cached alignment
        cached_msa = os.path.join(chain_dir, "cached_msa.fasta")
        
        # Check if cached alignment exists
        if os.path.exists(cached_msa):
            print(f"Using cached alignment for {chain_type} chain from {cached_msa}")
            return cached_msa
        
        print(f"No cached alignment found. Running MUSCLE for {chain_type} chain...")
        
        # Create new alignment
        temp_fasta = os.path.join(chain_dir, "input.fasta")
        aligned_file = os.path.join(chain_dir, "msa.fasta")
        
        try:
            # Create FASTA records
            seq_len = len(sequences[0])
            records = []
            for i, seq in enumerate(sequences):
                header = f">seq_{chain_type}_{i}/1-{seq_len}"
                record = SeqRecord(Seq(seq), id=header, description="")
                records.append(record)
            SeqIO.write(records, temp_fasta, "fasta")
            
            # Run MUSCLE
            muscle_exe = r"muscle-win64.v5.3.exe"
            cmd = [muscle_exe, "-super5", temp_fasta, "-output", aligned_file]
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"MUSCLE Error: {stderr}")
            
            # Process alignment
            with open(aligned_file, 'r') as f:
                aligned_content = f.read()
            
            # Update focus sequence header
            aligned_lines = aligned_content.split('\n')
            if aligned_lines[0].startswith('>'):
                aligned_lines[0] = f">focus_sequence/1-{seq_len}"
            
            # Save cache
            with open(cached_msa, 'w') as f:
                f.write('\n'.join(aligned_lines))
            
            # Cleanup
            for f in [temp_fasta, aligned_file]:
                if os.path.exists(f):
                    os.remove(f)
                    
            return cached_msa
            
        except Exception as e:
            print(f"Error in MSA processing: {str(e)}")
            # Cleanup on error
            for f in [temp_fasta, aligned_file]:
                if os.path.exists(f):
                    os.remove(f)
            raise
    
    def clear_alignment_cache(self):
        """Clear cached alignments"""
        for chain_type in ["heavy", "light"]:
            chain_dir = os.path.join(self.output_dir, chain_type)
            cached_msa = os.path.join(chain_dir, "cached_msa.fasta")
            if os.path.exists(cached_msa):
                os.remove(cached_msa)
                print(f"Cleared cached alignment for {chain_type} chain")

    def prepare_and_train_models(
            self, 
            train_path: str, 
            num_updates: int = 100000, 
            use_cache: bool = True,
            save_models: bool = True,
            heavy_model_timestamp: Optional[str] = None,
            light_model_timestamp: Optional[str] = None,
            train_heavy: bool = True,
            train_light: bool = True
        ) -> Tuple[Optional[DeepSequenceWrapper], Optional[DeepSequenceWrapper], List[str], List[str]]:
            """Prepare MSA, train/load models"""
            positive_heavy, positive_light = self.sequence_loader.load_sequences(train_path)
            
            if not use_cache and (train_heavy or train_light):
                self.clear_alignment_cache()
            
            heavy_model = None
            light_model = None
            
            # Heavy chain
            if train_heavy or heavy_model_timestamp:
                heavy_msa = self.process_sequences_to_msa(positive_heavy, "heavy")
                heavy_model = DeepSequenceWrapper(
                    alignment_file=heavy_msa,
                    working_dir=os.path.join(self.output_dir, "heavy")
                )
                
                if train_heavy:
                    print("Training new heavy chain model...")
                    heavy_model.train_model(num_updates=num_updates)
                    if save_models:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        heavy_model.model.save_parameters(f"heavy_chain_model_{timestamp}")
                        print(f"Heavy chain model saved with timestamp: {timestamp}")
                else:
                    print(f"Loading heavy chain model from timestamp: {heavy_model_timestamp}")
                    heavy_model.model = VariationalAutoencoder(
                        data=heavy_model.data,
                        working_dir=heavy_model.working_dir,
                        **heavy_model.model_params
                    ).to(heavy_model.device)
                    
                    load_success = heavy_model.model.load_parameters(f"heavy_chain_model_{heavy_model_timestamp}")
                    if not load_success:
                        print("Failed to load heavy chain model, training new one...")
                        heavy_model.train_model(num_updates=num_updates)
                        if save_models:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            heavy_model.model.save_parameters(f"heavy_chain_model_{timestamp}")
                            print(f"Heavy chain model saved with timestamp: {timestamp}")
            
            # Light chain
            if train_light or light_model_timestamp:
                light_msa = self.process_sequences_to_msa(positive_light, "light")
                light_model = DeepSequenceWrapper(
                    alignment_file=light_msa,
                    working_dir=os.path.join(self.output_dir, "light")
                )
                
                if train_light:
                    print("Training new light chain model...")
                    light_model.train_model(num_updates=num_updates)
                    if save_models:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        light_model.model.save_parameters(f"light_chain_model_{timestamp}")
                        print(f"Light chain model saved with timestamp: {timestamp}")
                else:
                    print(f"Loading light chain model from timestamp: {light_model_timestamp}")
                    light_model.model = VariationalAutoencoder(
                        data=light_model.data,
                        working_dir=light_model.working_dir,
                        **light_model.model_params
                    ).to(light_model.device)
                    
                    load_success = light_model.model.load_parameters(f"light_chain_model_{light_model_timestamp}")
                    if not load_success:
                        print("Failed to load light chain model, training new one...")
                        light_model.train_model(num_updates=num_updates)
                        if save_models:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            light_model.model.save_parameters(f"light_chain_model_{timestamp}")
                            print(f"Light chain model saved with timestamp: {timestamp}")
            
            return heavy_model, light_model, positive_heavy, positive_light

    def list_saved_models(self) -> Dict[str, List[str]]:
        """List all saved models with their timestamps
        
        Returns:
            Dictionary with 'heavy' and 'light' keys containing lists of timestamps
        """
        saved_models = {'heavy': [], 'light': []}
        
        for model_type in ['heavy', 'light']:
            params_dir = os.path.join(self.output_dir, model_type, 'params')
            if os.path.exists(params_dir):
                for filename in os.listdir(params_dir):
                    if filename.startswith(f"{model_type}_chain_model_") and filename.endswith("_params.pkl"):
                        timestamp = filename.replace(f"{model_type}_chain_model_", "").replace("_params.pkl", "")
                        saved_models[model_type].append(timestamp)
                saved_models[model_type].sort(reverse=True)  # Most recent first
        
        return saved_models

    def generate_paired_negatives(
        self, 
        heavy_model: DeepSequenceWrapper,
        light_model: DeepSequenceWrapper,
        positive_heavy: List[str],
        positive_light: List[str],
        n_pairs: int = 50000,
        pair_types: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Generate paired negative samples"""
        random.seed(42)  # Reset seed for reproducibility
        
        if pair_types is None:
            pair_types = {
                'both_negative': 0.4,
                'negative_heavy': 0.3,
                'negative_light': 0.3
            }
        
        type_counts = {
            ptype: int(n_pairs * prop)
            for ptype, prop in pair_types.items()
        }
        
        # Generate negative samples
        negative_heavy = heavy_model.generate_negative_samples(
            n_samples=n_pairs,
            temp=0.8,
            min_dist=2.0
        )
        
        negative_light = light_model.generate_negative_samples(
            n_samples=n_pairs,
            temp=0.8,
            min_dist=2.0
        )
        
        # Generate pairs
        paired_sequences = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate each type of pair
        for pair_type, count in type_counts.items():
            for i in range(count):
                if pair_type == 'both_negative':
                    heavy_seq = random.choice(negative_heavy)
                    light_seq = random.choice(negative_light)
                    sources = ('negative', 'negative')
                elif pair_type == 'negative_heavy':
                    heavy_seq = random.choice(negative_heavy)
                    light_seq = random.choice(positive_light)
                    sources = ('negative', 'positive')
                else:  # negative_light
                    heavy_seq = random.choice(positive_heavy)
                    light_seq = random.choice(negative_light)
                    sources = ('positive', 'negative')
                
                pair = {
                    'pair_id': f'{pair_type}_{timestamp}_{i}',
                    'AA_Sequence': f"{heavy_seq}</s>{light_seq}",
                    'heavy_chain': heavy_seq,
                    'light_chain': light_seq,
                    'heavy_source': sources[0],
                    'light_source': sources[1],
                    'pair_type': pair_type
                }
                paired_sequences.append(pair)
        
        # Create DataFrame
        df_pairs = pd.DataFrame(paired_sequences)
        df_pairs = df_pairs[['AA_Sequence', 'heavy_chain', 'light_chain', 
                            'pair_id', 'heavy_source', 'light_source', 'pair_type']]
        return df_pairs
    
    def save_to_csv(self, df_pairs: pd.DataFrame, filename: Optional[str] = None) -> str:
        """Save paired sequences to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"paired_negatives_{timestamp}.csv"
        
        output_path = os.path.join(self.output_dir, filename)
        df_pairs.to_csv(output_path, index=False)
        print(f"Saved paired sequences to {output_path}")
        
        return output_path

def main():
    try:
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        
        train_path = "C:/Users/jovin/OneDrive - Nanyang Technological University/FYP/WIP/vae_sequences.csv"
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data file not found: {train_path}")
        
        output_dir = "./antibody_output"
        os.makedirs(output_dir, exist_ok=True)
        
        generator = PairedNegativeGenerator(output_dir=output_dir)
        
        saved_models = generator.list_saved_models()
        
        print("\nHeavy Chain Model:")
        if saved_models['heavy']:
            print("Found saved heavy chain models:", saved_models['heavy'])
            use_saved_heavy = input("Use most recent saved heavy chain model? (y/n): ").lower() == 'y'
            heavy_timestamp = saved_models['heavy'][0] if use_saved_heavy else None
            train_heavy = not use_saved_heavy
        else:
            print("No saved heavy chain models found.")
            train_heavy = True
            heavy_timestamp = None
            
        print("\nLight Chain Model:")
        if saved_models['light']:
            print("Found saved light chain models:", saved_models['light'])
            use_saved_light = input("Use most recent saved light chain model? (y/n): ").lower() == 'y'
            light_timestamp = saved_models['light'][0] if use_saved_light else None
            train_light = not use_saved_light
        else:
            print("No saved light chain models found.")
            train_light = True
            light_timestamp = None

        heavy_model, light_model, positive_heavy, positive_light = generator.prepare_and_train_models(
            train_path=train_path,
            num_updates=200000,
            use_cache=True,
            save_models=True, 
            heavy_model_timestamp=heavy_timestamp,
            light_model_timestamp=light_timestamp,
            train_heavy=train_heavy,
            train_light=train_light
        )

        print("\nGenerating paired negative sequences...")
        paired_sequences_df = generator.generate_paired_negatives(
            heavy_model=heavy_model,
            light_model=light_model,
            positive_heavy=positive_heavy,
            positive_light=positive_light,
            n_pairs=75000,
            pair_types={
                'both_negative': 0.4,
                'negative_heavy': 0.3,
                'negative_light': 0.3
            }
        )
        
        csv_path = generator.save_to_csv(paired_sequences_df)
        
        print("\nGeneration Summary:")
        print("-" * 50)
        print("\nPair Type Distribution:")
        type_counts = paired_sequences_df['pair_type'].value_counts()
        for pair_type, count in type_counts.items():
            print(f"{pair_type}: {count} pairs ({count/len(paired_sequences_df)*100:.1f}%)")
        
        print(f"\nTotal pairs generated: {len(paired_sequences_df)}")
        print(f"Results saved to: {csv_path}")
        
        print("\nExample Generated Pairs:")
        print("-" * 50)
        for i, row in paired_sequences_df.head(3).iterrows():
            print(f"\nPair {i+1} ({row['pair_type']}):")
            print(f"Heavy Chain ({row['heavy_source']}): {row['heavy_chain'][:50]}...")
            print(f"Light Chain ({row['light_source']}): {row['light_chain'][:50]}...")
        
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("Please check the path to your training data file.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
    finally:
        print("\nProcess completed.")

if __name__ == "__main__":
    main()
