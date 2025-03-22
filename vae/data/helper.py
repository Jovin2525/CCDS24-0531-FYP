import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import os
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm


class DataHelper:
    def __init__(
        self,
        dataset: str = "",
        alignment_file: str = "",
        focus_seq_name: str = "",
        calc_weights: bool = True,
        working_dir: str = ".",
        theta: float = 0.2,
        load_all_sequences: bool = True,
        alphabet_type: str = "protein",
        batch_size: int = 1000  # Added batch size for memory efficiency
    ):
        """Initialize with memory-efficient processing"""
        # Original initialization
        np.random.seed(42)
        self.dataset = dataset
        self.alignment_file = alignment_file
        self.focus_seq_name = focus_seq_name
        self.working_dir = working_dir
        self.calc_weights = calc_weights
        self.alphabet_type = alphabet_type
        self.theta = theta
        self.load_all_sequences = load_all_sequences
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize ELBO
        self.wt_elbo = None

        # Configure datasets and alphabet
        if self.dataset:
            self.configure_datasets()

        self._setup_alphabet()
        
        # Process alignment
        self.gen_basic_alignment()
        if self.load_all_sequences:
            self.gen_full_alignment()

    def _setup_alphabet(self):
        """Configure alphabet based on type"""
        alphabets = {
            "protein": ("ACDEFGHIKLMNPQRSTVWY", "DEKRHNQSTPGAVILMCFYW"),
            "RNA": ("ACGU", "ACGU"),
            "DNA": ("ACGT", "ACGT"),
            "allelic": ("012", "012")
        }
        self.alphabet, self.reorder_alphabet = alphabets.get(
            self.alphabet_type, 
            alphabets["protein"]
        )

    def _batch_process(self, data: torch.Tensor, batch_size: int, func) -> List:
        """Process data in batches to save memory"""
        results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_result = func(batch)
            results.extend(batch_result)
        return results

    def gen_full_alignment(self):
        """Generate full alignment with optimized memory usage"""
        # Process sequences first
        print("Processing sequences...")
        processed_sequences = {}
        alphabet_set = set(self.alphabet)
        
        for seq_name, sequence in tqdm(self.seq_name_to_sequence.items()):
            sequence = sequence.replace(".", "-")
            processed_seq = [sequence[ix].upper() for ix in self.focus_cols]
            
            # Filter invalid sequences
            if all(letter in alphabet_set or letter == "-" for letter in processed_seq):
                processed_sequences[seq_name] = processed_seq

        # Initialize training data
        print("Encoding sequences...")
        num_sequences = len(processed_sequences)
        self.x_train = np.zeros(
            (num_sequences, len(self.focus_cols), len(self.alphabet)),
            dtype=np.float32  # Use float32 for memory efficiency
        )
        self.x_train_name_list = []

        # Encode sequences
        for i, (seq_name, sequence) in enumerate(tqdm(processed_sequences.items())):
            self.x_train_name_list.append(seq_name)
            for j, letter in enumerate(sequence):
                if letter in self.aa_dict:
                    self.x_train[i, j, self.aa_dict[letter]] = 1.0

        if self.calc_weights:
            print("Computing sequence weights...")
            self._compute_weights_batched()
        else:
            self.weights = np.ones(self.x_train.shape[0], dtype=np.float32)

        self.Neff = np.sum(self.weights)
        print(f"Neff = {self.Neff}")
        print(f"Data Shape = {self.x_train.shape}")

    def _compute_weights_batched(self):
        """Compute sequence weights using batched processing"""
        x_flat = torch.tensor(
            self.x_train.reshape(self.x_train.shape[0], -1),
            dtype=torch.float32,
            device=self.device
        )
        num_sequences = len(x_flat)
        weights = torch.zeros(num_sequences, device=self.device)
        
        # Compute norms once
        norms = torch.norm(x_flat, dim=1)
        
        # Process in batches
        for i in tqdm(range(0, num_sequences, self.batch_size)):
            batch_end = min(i + self.batch_size, num_sequences)
            batch_x = x_flat[i:batch_end]
            
            # Calculate similarities for this batch
            batch_norms = norms[i:batch_end].unsqueeze(1)
            similarities = torch.mm(batch_x, x_flat.t()) / (
                batch_norms @ norms.unsqueeze(0)
            )
            
            # Count sequences above threshold
            weights[i:batch_end] = torch.sum(similarities > 1 - self.theta, dim=1)
            
            # Clear GPU memory
            del similarities
            torch.cuda.empty_cache()
        
        # Convert to weights
        self.weights = (1.0 / weights).cpu().numpy()

    def custom_mutant_matrix(
        self, 
        input_filename: str,
        model,
        N_pred_iterations: int = 10,
        minibatch_size: int = 2000,
        filename_prefix: str = "",
        offset: int = 0
    ) -> Optional[Tuple[List[str], np.ndarray]]:
        """Optimized mutation effect prediction"""
        # Process mutations in batches
        mutant_data = self._process_mutations(input_filename)
        
        # Generate one-hot encodings efficiently
        one_hot_encodings = self._generate_one_hot(mutant_data)
        
        # Calculate predictions in batches
        predictions = self._batch_predict(
            model,
            one_hot_encodings,
            N_pred_iterations,
            minibatch_size
        )
        
        return self._process_predictions(predictions, mutant_data, filename_prefix)
    
    def gen_basic_alignment(self):
            """Read training alignment and store basics in class instance"""
            # Create amino acid dictionaries
            self.aa_dict = {aa: i for i, aa in enumerate(self.alphabet)}
            self.num_to_aa = {i: aa for aa, i in self.aa_dict.items()}

            # Get reordering indices
            ix = np.array([self.alphabet.find(s) for s in self.reorder_alphabet])

            # Initialize sequence storage
            self.seq_name_to_sequence = defaultdict(str)
            self.seq_names = []

            # Read alignment file
            name = ""
            try:
                with open(self.alignment_file, "r") as input_file:
                    for line in input_file:
                        line = line.rstrip()
                        if line.startswith(">"):
                            name = line
                            self.seq_names.append(name)
                        else:
                            self.seq_name_to_sequence[name] += line
            except FileNotFoundError:
                raise FileNotFoundError(f"Alignment file not found: {self.alignment_file}")
            except Exception as e:
                raise Exception(f"Error reading alignment file: {str(e)}")

            # Set focus sequence if not provided
            if not self.focus_seq_name:
                self.focus_seq_name = self.seq_names[0]

            # Process focus sequence
            self.focus_seq = self.seq_name_to_sequence[self.focus_seq_name]
            if not self.focus_seq:
                raise ValueError(f"Focus sequence {self.focus_seq_name} not found in alignment")

            # Get focus columns (uppercase letters)
            self.focus_cols = [ix for ix, s in enumerate(self.focus_seq) if s == s.upper()]
            if not self.focus_cols:
                raise ValueError("No uppercase letters found in focus sequence")

            # Extract trimmed sequence
            self.focus_seq_trimmed = [self.focus_seq[ix] for ix in self.focus_cols]
            self.seq_len = len(self.focus_cols)
            self.alphabet_size = len(self.alphabet)

            # Process focus sequence location
            try:
                focus_loc = self.focus_seq_name.split("/")[-1]
                start, stop = focus_loc.split("-")
                self.focus_start_loc = int(start)
                self.focus_stop_loc = int(stop)
            except (IndexError, ValueError):
                raise ValueError(f"Invalid focus sequence name format: {self.focus_seq_name}")

            # Create lookup dictionaries
            self.uniprot_focus_cols_list = [
                idx_col + self.focus_start_loc for idx_col in self.focus_cols
            ]
            self.uniprot_focus_col_to_wt_aa_dict = {
                idx_col + self.focus_start_loc: self.focus_seq[idx_col]
                for idx_col in self.focus_cols
            }
            self.uniprot_focus_col_to_focus_idx = {
                idx_col + self.focus_start_loc: idx_col 
                for idx_col in self.focus_cols
            }
            
    def get_pattern_activations(
        self,
        model,
        update_num: int,
        filename_prefix: str = "",
        verbose: bool = False,
        minibatch_size: int = 2000
    ):
        """Get pattern activations with memory optimization"""
        activations_filename = os.path.join(
            self.working_dir,
            "embeddings",
            f"{filename_prefix}_pattern_activations.csv"
        )
        
        os.makedirs(os.path.dirname(activations_filename), exist_ok=True)
        
        with open(activations_filename, "w") as output_file:
            for batch_start in range(0, len(self.x_train_name_list), minibatch_size):
                batch_end = min(batch_start + minibatch_size, len(self.x_train_name_list))
                batch_seqs = torch.tensor(
                    self.x_train[batch_start:batch_end],
                    dtype=torch.float32,
                    device=self.device
                )
                
                # Get activations for batch
                with torch.no_grad():
                    batch_activation = model.get_pattern_activations(batch_seqs)
                
                # Process and write results
                for j, idx in enumerate(range(batch_start, batch_end)):
                    activation_values = batch_activation[j].cpu().tolist()
                    sample_name = self.x_train_name_list[idx]
                    output_line = [str(update_num), sample_name] + [
                        f"{val:.6f}" for val in activation_values
                    ]
                    
                    if verbose:
                        print("\t".join(output_line))
                    output_file.write(",".join(output_line) + "\n")
                
                # Clear GPU memory
                del batch_seqs, batch_activation
                torch.cuda.empty_cache()

    def get_elbo_samples(
        self,
        model,
        N_pred_iterations: int = 100,
        minibatch_size: int = 2000
    ):
        """Get ELBO samples with memory optimization"""
        num_sequences = self.one_hot_mut_array_with_wt.shape[0]
        self.prediction_matrix = np.zeros(
            (num_sequences, N_pred_iterations),
            dtype=np.float32
        )
        
        for iteration in range(N_pred_iterations):
            batch_order = np.random.permutation(num_sequences)
            
            for batch_start in range(0, num_sequences, minibatch_size):
                batch_end = min(batch_start + minibatch_size, num_sequences)
                batch_indices = batch_order[batch_start:batch_end]
                
                # Process batch
                batch_data = torch.tensor(
                    self.one_hot_mut_array_with_wt[batch_indices],
                    dtype=torch.float32,
                    device=self.device
                )
                
                with torch.no_grad():
                    batch_preds = model.all_likelihood_components(batch_data)
                
                # Store results
                for k, idx_batch in enumerate(batch_indices):
                    self.prediction_matrix[idx_batch, iteration] = (
                        batch_preds[k].cpu().item()
                    )
                
                # Clear GPU memory
                del batch_data, batch_preds
                torch.cuda.empty_cache()