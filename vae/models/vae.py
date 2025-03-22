from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from scipy.special import erfinv
from typing import List, Tuple, Optional, Union
import math


class VariationalAutoencoder(nn.Module):
    def __init__(
        self,
        data,
        encoder_architecture: List[int] = [1500, 1500],
        decoder_architecture: List[int] = [100, 500],
        n_latent: int = 2,
        n_patterns: int = 4,
        batch_size: int = 100,
        encode_nonlinearity_type: str = "relu",
        decode_nonlinearity_type: str = "relu",
        final_decode_nonlinearity: str = "sigmoid",
        sparsity: Union[str, bool] = "logit",
        global_scale: float = 1.0,
        logit_p: float = 0.01,
        logit_sigma: float = 4.0,
        pattern_sigma: float = 1.0,
        warm_up: float = 0.0,
        convolve_encoder: bool = False,
        convolve_patterns: bool = True,
        conv_decoder_size: int = 10,
        conv_encoder_size: int = 10,
        output_bias: bool = True,
        final_pwm_scale: bool = False,
        working_dir: str = ".",
        learning_rate: float = 0.001,
        kl_scale: float = 1.0,
        b1: float = 0.9,
        b2: float = 0.999,
        random_seed: int = 42
    ):
        """Initialize VAE model"""
        super().__init__()

        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Save parameters
        self.working_dir = working_dir
        self.n_latent = n_latent
        self.encoder_architecture = encoder_architecture
        self.decoder_architecture = decoder_architecture
        self.n_patterns = n_patterns
        self.batch_size = batch_size

        # Modify decoder architecture for patterns
        if self.decoder_architecture:
            self.decoder_architecture[-1] = self.decoder_architecture[-1] * self.n_patterns
        else:
            self.n_latent = n_latent * self.n_patterns

        # Data parameters
        self.seq_len = data.seq_len
        self.alphabet_size = data.alphabet_size
        
        # Model options
        self.encode_nonlinearity_type = encode_nonlinearity_type
        self.decode_nonlinearity_type = decode_nonlinearity_type
        self.final_decode_nonlinearity = final_decode_nonlinearity
        self.convolve_encoder = convolve_encoder
        self.convolve_patterns = convolve_patterns
        self.conv_encoder_size = conv_encoder_size
        self.conv_decoder_size = conv_decoder_size
        self.output_bias = output_bias
        self.final_pwm_scale = final_pwm_scale
        
        # Training parameters
        self.epsilon = 1e-8 if torch.get_default_dtype() != torch.float16 else 1e-6
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.warm_up = torch.tensor(warm_up, device=self.device)
        self.kl_scale = torch.tensor(kl_scale, device=self.device)

        # Sparsity parameters
        self.sparsity = sparsity
        self.global_scale = torch.tensor(global_scale, device=self.device)
        self.inv_global_scale = torch.tensor(1.0 / global_scale, device=self.device)
        self.logit_p = logit_p
        self.logit_mu = torch.tensor(
            np.sqrt(2.0) * logit_sigma * erfinv(2.0 * logit_p - 1.0), 
            device=self.device
        )
        self.logit_sigma = torch.tensor(logit_sigma, device=self.device)
        self.pattern_sigma = pattern_sigma

        # Initialize networks
        self._init_encoder()
        self._init_decoder()
        
        # Adam optimizer parameters
        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.t = 0  # Step counter
        
        # Setup optimizer with proper state initialization
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Initialize optimizer state
        self.optimizer.state = defaultdict(dict)
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state[p]
                state['step'] = torch.zeros(1, device=self.device)
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

        self.t = 0  # Step counter
        self.to(self.device)

    def _init_encoder(self):
        """Initialize encoder network"""
        layers = []
        input_size = self.seq_len * self.alphabet_size

        # Add convolutional layer if needed
        if self.convolve_encoder:
            self.conv_encode = nn.Linear(
                self.alphabet_size,
                self.conv_encoder_size
            )
            input_size = self.seq_len * self.conv_encoder_size

        # Add encoder layers
        for hidden_size in self.encoder_architecture:
            layers.append(nn.Linear(input_size, hidden_size))
            if self.encode_nonlinearity_type == "relu":
                layers.append(nn.ReLU())
            elif self.encode_nonlinearity_type == "tanh":
                layers.append(nn.Tanh())
            elif self.encode_nonlinearity_type == "sigmoid":
                layers.append(nn.Sigmoid())
            input_size = hidden_size

        self.encoder = nn.Sequential(*layers)
        
        # Latent space projections
        self.fc_mu = nn.Linear(input_size, self.n_latent)
        self.fc_logvar = nn.Linear(input_size, self.n_latent)

    def _init_decoder(self):
        """Initialize decoder network with all variants"""
        layers = []
        input_size = self.n_latent

        # Add decoder layers
        for hidden_size in self.decoder_architecture:
            layers.append(nn.Linear(input_size, hidden_size))
            if self.decode_nonlinearity_type == "relu":
                layers.append(nn.ReLU())
            elif self.decode_nonlinearity_type == "tanh":
                layers.append(nn.Tanh())
            elif self.decode_nonlinearity_type == "sigmoid":
                layers.append(nn.Sigmoid())
            input_size = hidden_size

        self.decoder = nn.Sequential(*layers)
        
        # Set final output size
        self.final_output_size = (
            self.decoder_architecture[-1] 
            if self.decoder_architecture 
            else self.n_latent
        )

        # Initialize output layers
        if self.convolve_patterns:
            # Convolution weights
            self.conv_decode = nn.Linear(
                self.conv_decoder_size,
                self.alphabet_size
            )
            # Output weights
            self.W_out = nn.Linear(
                self.final_output_size,
                self.seq_len * self.conv_decoder_size
            )
        else:
            # Direct output weights
            self.W_out = nn.Linear(
                self.final_output_size,
                self.seq_len * self.alphabet_size
            )

        # Optional output bias
        if self.output_bias:
            self.b_out = nn.Parameter(
                torch.zeros(self.seq_len * self.alphabet_size)
            )

        # Sparsity parameters
        if self.sparsity:
            self.W_out_scale = nn.Parameter(
                torch.zeros(
                    self.final_output_size // self.n_patterns,
                    self.seq_len
                )
            )

        # PWM scale
        if self.final_pwm_scale:
            self.final_pwm_scale_param = nn.Parameter(torch.ones(1))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space"""
        batch_size, seq_len, alphabet_size = x.shape

        if self.convolve_encoder:
            x_flat = x.reshape(-1, alphabet_size)
            x_conv = self._encode_nonlinearity(self.conv_encode(x_flat))
            x = x_conv.reshape(batch_size, -1)
        else:
            x = x.reshape(batch_size, -1)

        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor, x: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode latent representation"""
        h = self.decoder(z)
        
        # Apply sparsity if needed
        if self.sparsity:
            if self.sparsity == "logit":
                scale = torch.sigmoid(self.W_out_scale)
            else:
                scale = torch.exp(self.W_out_scale)
            scale = scale.repeat(self.n_patterns, 1, 1)

        # Get output weights
        if self.convolve_patterns:
            W = self.W_out.weight.reshape(
                self.final_output_size,
                self.seq_len,
                self.conv_decoder_size
            )
            if self.sparsity:
                W = W * scale
            W = self.conv_decode(W)
        else:
            W = self.W_out.weight.reshape(
                self.final_output_size,
                self.seq_len,
                self.alphabet_size
            )
            if self.sparsity:
                W = W * scale

        # Reshape for multiplication
        W = W.reshape(self.final_output_size, -1)
        h_flat = torch.matmul(h, W)
        
        if self.output_bias:
            h_flat = h_flat + self.b_out

        if self.final_pwm_scale:
            h_flat = h_flat * F.softplus(self.final_pwm_scale_param)

        # Reshape and apply softmax
        logits = h_flat.reshape(-1, self.seq_len, self.alphabet_size)
        probs = F.softmax(logits, dim=-1)
        
        # Calculate log probabilities if input provided
        if x is not None:
            log_probs = F.log_softmax(logits, dim=-1)
            logpxz = torch.sum(x * log_probs, dim=(1, 2))
        else:
            logpxz = None

        return probs, logpxz, h

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x, logpxz, h = self.decode(z, x)
        return recon_x, logpxz, mu, logvar, z

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def KLD_diag_gaussians(self, mu: torch.Tensor, log_sigma: torch.Tensor, 
                          prior_mu: torch.Tensor, prior_log_sigma: torch.Tensor) -> torch.Tensor:
        """KL divergence between two Diagonal Gaussians"""
        return prior_log_sigma - log_sigma + 0.5 * (
            torch.exp(2. * log_sigma) + (mu - prior_mu).pow(2)
        ) * torch.exp(-2. * prior_log_sigma) - 0.5

    def _anneal(self, update_num: int) -> torch.Tensor:
        """Anneal the KL if using annealing"""
        # Convert update_num to tensor
        update_num = torch.tensor(update_num, dtype=torch.float32, device=self.device)
        
        # If warm_up is not a tensor, convert it
        if not isinstance(self.warm_up, torch.Tensor):
            self.warm_up = torch.tensor(self.warm_up, dtype=torch.float32, device=self.device)
            
        # Return annealing factor
        return torch.where(
            update_num < self.warm_up,
            update_num / self.warm_up,
            torch.ones_like(update_num, device=self.device)
        )

    def update(self, x: torch.Tensor, Neff: float, step: int) -> Tuple[float, float, float, float]:
        """Update model parameters with fixed optimizer step"""
        self.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        recon_x, logpxz, mu, logvar, z = self(x)
        
        # Calculate losses
        KLD_latent = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        KLD_params = self.gen_kld_params()
        
        if self.sparsity:
            KLD_params += self.gen_kld_sparsity(self.sparsity)
        
        # Total loss
        warm_up_scale = self._anneal(step)
        loss = -torch.mean(logpxz - (warm_up_scale * self.kl_scale * KLD_latent)) - (
            warm_up_scale * KLD_params / Neff
        )
        
        # Backward pass
        loss.backward()
        
        # Manual optimizer step to ensure proper state handling
        with torch.no_grad():
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                        
                    # Get state
                    state = self.optimizer.state[p]
                    
                    # Update step
                    state['step'] += 1
                    
                    # Update biased first moment estimate
                    state['exp_avg'].mul_(group['betas'][0]).add_(
                        p.grad, alpha=1 - group['betas'][0]
                    )
                    
                    # Update biased second raw moment estimate
                    state['exp_avg_sq'].mul_(group['betas'][1]).addcmul_(
                        p.grad, p.grad, value=1 - group['betas'][1]
                    )
                    
                    # Bias correction
                    bias_correction1 = 1 - group['betas'][0] ** state['step'].item()
                    bias_correction2 = 1 - group['betas'][1] ** state['step'].item()
                    
                    # Compute step size
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                    
                    # Update parameters
                    p.addcdiv_(
                        state['exp_avg'],
                        state['exp_avg_sq'].sqrt().add_(group['eps']),
                        value=-step_size
                    )
        
        self.t += 1
        
        return (
            -loss.item(),
            torch.mean(logpxz).item(),
            KLD_params.item(),
            torch.mean(KLD_latent).item()
        )

    def save_parameters(self, filename: str):
        """Save model parameters with proper state handling"""
        try:
            import pickle
            
            save_path = os.path.join(self.working_dir, "params")
            os.makedirs(save_path, exist_ok=True)
            
            temp_path = os.path.join(save_path, f"{filename}_params.tmp.pkl")
            final_path = os.path.join(save_path, f"{filename}_params.pkl")
            
            # Convert state dict properly
            model_state = {}
            for name, param in self.state_dict().items():
                if isinstance(param, torch.Tensor):
                    model_state[name] = param.cpu().detach().numpy()
                else:
                    model_state[name] = param
                    
            # Convert optimizer state properly
            optimizer_state = {}
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    state = self.optimizer.state[p]
                    state_dict = {}
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            state_dict[key] = value.cpu().detach().numpy()
                        else:
                            state_dict[key] = value
                    optimizer_state[id(p)] = state_dict
            
            checkpoint = {
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer_state,
                'step': self.t,
            }
            
            with open(temp_path, 'wb') as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            if os.path.exists(temp_path):
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.rename(temp_path, final_path)
                
        except Exception as e:
            print(f"Error saving parameters: {str(e)}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise

    def recognize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get latent representation of input"""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x)
        return mu.cpu().numpy(), logvar.cpu().numpy()


    def load_parameters(self, filename: str) -> bool:
        """Load model parameters
        
        Args:
            filename: Name of the parameter file to load
            
        Returns:
            bool: True if parameters were loaded successfully
        """
        try:
            import pickle
            
            # Construct file path
            load_path = os.path.join(self.working_dir, "params", f"{filename}_params.pkl")
            
            if not os.path.exists(load_path):
                print(f"Parameter file not found: {load_path}")
                return False
            
            # Load checkpoint
            with open(load_path, 'rb') as f:
                checkpoint = pickle.load(f)
                
            # Convert numpy arrays back to tensors for model state
            model_state = {}
            for name, param in checkpoint['model_state_dict'].items():
                if isinstance(param, np.ndarray):
                    model_state[name] = torch.from_numpy(param).to(self.device)
                else:
                    model_state[name] = param
            
            # Load model state
            self.load_state_dict(model_state)
            
            # Reset optimizer
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                betas=(self.b1, self.b2)
            )
            
            # Convert optimizer state dict
            optimizer_state = checkpoint['optimizer_state_dict']
            new_optimizer_state = {}
            
            # Map state to new parameter references
            param_mapping = {name: param for name, param in self.named_parameters()}
            
            for param_id, state in optimizer_state.items():
                for key, value in state.items():
                    if isinstance(value, np.ndarray):
                        state[key] = torch.from_numpy(value).to(self.device)
                        
            # Create new state dict structure
            optimizer_state_dict = {
                'state': optimizer_state,
                'param_groups': [{
                    'lr': self.learning_rate,
                    'betas': (self.b1, self.b2),
                    'eps': 1e-8,
                    'weight_decay': 0,
                    'amsgrad': False,
                    'maximize': False,
                    'foreach': None,
                    'capturable': False,
                    'params': [id(p) for p in self.parameters()]
                }]
            }
            
            # Load optimizer state
            self.optimizer.load_state_dict(optimizer_state_dict)
            
            # Load step counter
            self.t = checkpoint['step']
            
            return True
            
        except Exception as e:
            print(f"Error loading parameters: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def gen_kld_params(self) -> torch.Tensor:
            """Generate KL divergence for parameters"""
            KLD_params = torch.zeros(1, device=self.device)
            
            # Add KL for all decoder weights
            for name, param in self.named_parameters():
                if 'decode' in name:
                    KLD_params += 0.5 * torch.sum(param.pow(2))
            
            # Add KL for output parameters
            if hasattr(self, 'W_out'):
                KLD_params += 0.5 * torch.sum(self.W_out.weight.pow(2))
            if hasattr(self, 'b_out'):
                KLD_params += 0.5 * torch.sum(self.b_out.pow(2))
            if hasattr(self, 'final_pwm_scale_param'):
                KLD_params += 0.5 * torch.sum(self.final_pwm_scale_param.pow(2))
                
            return KLD_params

    def gen_kld_sparsity(self, sparsity: str) -> torch.Tensor:
        """Generate KL divergence for sparsity parameters"""
        if not hasattr(self, 'W_out_scale'):
            return torch.tensor(0.0, device=self.device)

        if sparsity == "logit":
            # Continuous relaxation of spike and slab prior
            KLD_fadeout = -self.KLD_diag_gaussians(
                self.W_out_scale.flatten(),
                torch.zeros_like(self.W_out_scale.flatten()),
                self.logit_mu,
                torch.log(self.logit_sigma)
            )
        elif sparsity == "analytic":
            # Moment-matched Gaussian approximation
            KLD_fadeout = -self.KLD_diag_gaussians(
                self.W_out_scale.flatten(),
                torch.zeros_like(self.W_out_scale.flatten()),
                torch.log(self.global_scale),
                torch.log(torch.tensor(np.pi / 2, device=self.device))
            )
        else:
            # Sample-based KL estimation
            W_scale = torch.exp(self.W_out_scale)
            
            if sparsity == "horseshoe":
                # Half-Cauchy hyperprior
                KLD_fadeout = (
                    torch.log(torch.tensor(2.0, device=self.device)) +
                    torch.log(self.global_scale) -
                    torch.log(torch.tensor(np.pi, device=self.device)) +
                    torch.log(W_scale) -
                    torch.log(self.global_scale.pow(2) + W_scale.pow(2))
                )
            elif sparsity == "laplacian":
                # Exponential hyperprior
                KLD_fadeout = (
                    torch.log(torch.tensor(2.0, device=self.device)) +
                    torch.log(self.inv_global_scale) -
                    self.inv_global_scale * W_scale.pow(2) +
                    2.0 * torch.log(W_scale)
                )
            elif sparsity == "ard":
                # Inverse-Gamma hyperprior
                KLD_fadeout = (
                    torch.log(torch.tensor(2.0, device=self.device)) +
                    self.global_scale * torch.log(self.global_scale) -
                    torch.lgamma(self.global_scale) -
                    (self.global_scale / (W_scale.pow(2) + self.epsilon)) -
                    (2.0 * self.global_scale * torch.log(W_scale))
                )
            else:
                raise ValueError(f"Unknown sparsity type: {sparsity}")

        return torch.sum(KLD_fadeout)

    def get_pattern_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get pattern activations for sequences"""
        self.eval()
        with torch.no_grad():
            # Get encoder output
            if self.convolve_encoder:
                x_flat = x.reshape(-1, self.alphabet_size)
                x_conv = self._encode_nonlinearity(self.conv_encode(x_flat))
                x_reshaped = x_conv.reshape(x.size(0), -1)
            else:
                x_reshaped = x.reshape(x.size(0), -1)
            
            # Get activations
            activations = self.encoder(x_reshaped)
            if self.encode_nonlinearity_type == "relu":
                activations = F.relu(activations)
            elif self.encode_nonlinearity_type == "tanh":
                activations = torch.tanh(activations)
            elif self.encode_nonlinearity_type == "sigmoid":
                activations = torch.sigmoid(activations)
            
        return activations

    def all_likelihood_components(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate all likelihood components"""
        self.eval()
        with torch.no_grad():
            # Forward pass
            recon_x, logpxz, mu, logvar, z = self(x)
            
            # Calculate KL divergence
            KLD_latent = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp(),
                dim=1
            )
            
            # Calculate ELBO
            elbo = logpxz - KLD_latent
            
        return elbo

    def decode_sparse(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode with sparsity"""
        # Decoder layers
        for i, layer in enumerate(self.decoder):
            h = layer(h)
            if i < len(self.decoder) - 1:  # Not last layer
                h = self._decode_nonlinearity(h)

        # Apply sparsity
        W_out = self.W_out.weight
        if self.sparsity:
            W_scale = self.W_out_scale
            if self.sparsity == "logit":
                W_scale = torch.sigmoid(W_scale)
            else:
                W_scale = torch.exp(W_scale)
            W_scale = W_scale.repeat(self.n_patterns, 1, 1)
            W_out = W_out.reshape(self.final_output_size, self.seq_len, -1) * W_scale
            W_out = W_out.reshape(self.final_output_size, -1)

        # Output layer
        if self.convolve_patterns:
            if hasattr(self, 'conv_decode'):
                W_out = W_out.reshape(-1, self.conv_decoder_size)
                W_out = self.conv_decode(W_out)
                W_out = W_out.reshape(self.final_output_size, -1)

        # Add bias if needed
        output = torch.matmul(h, W_out)
        if self.output_bias:
            output = output + self.b_out

        # Apply PWM scale if needed
        if self.final_pwm_scale:
            output = output * F.softplus(self.final_pwm_scale_param)

        # Reshape and apply softmax
        output = output.reshape(-1, self.seq_len, self.alphabet_size)
        probs = F.softmax(output, dim=-1)
        
        # Calculate log probabilities
        log_probs = F.log_softmax(output, dim=-1)
        logpxz = torch.sum(x * log_probs, dim=(1, 2))

        return probs, logpxz, h

    def _dropout(self, x: torch.Tensor, p: float = 0.5, training: bool = False) -> torch.Tensor:
        """Apply dropout during training"""
        if training and p > 0:
            mask = torch.bernoulli(torch.ones_like(x) * (1 - p))
            return x * mask / (1 - p)
        return x