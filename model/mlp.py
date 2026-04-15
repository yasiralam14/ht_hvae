import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    """
    Optimized MLP for HVAE Encoder q(z|x).
    1. Removes bottlenecks (Maintains width).
    2. Uses GELU (Matches BERT/Transformer activations).
    3. Implements Near-Zero Initialization to prevent early KL shock.
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        # 1. Maintain Width: Do not compress to //2 or //4 immediately.
        # We want deep non-linearities, not compression.
        hidden_dim = input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, 2 * latent_dim, bias=False)

        # Match activation to DistilBERT/GPT-2 (GELU is smoother than ReLU)
        self.activation = nn.GELU()

        # --- CRITICAL: Near-Zero Initialization ---
        # This ensures that at Step 0, the posterior q(z|x) is very close to N(0,1).
        # This prevents the initial KL loss from being huge, which scares the
        # optimizer into "killing" the latent variable immediately (Collapse).
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=0.001)


    def forward(self, h):
        # Post-Norm architecture (standard for Transformers)
        x = self.fc1(h)
        x = self.ln1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = self.activation(x)

        output = self.fc_out(x)

        mu, raw_var_score = output.chunk(2, dim=-1)

        # Robust Softplus
        sigma2 = F.softplus(raw_var_score) + 1e-6

        return mu, sigma2
