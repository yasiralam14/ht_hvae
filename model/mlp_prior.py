import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNetworkForPrior(nn.Module):
    """
    Optimized Prior Network p(z_i | z_t, z_i-1).
    1. Widened layers (No bottleneck).
    2. Context Dropout (Forces reliance on z_t).
    3. Near-Zero Init (Prevents initial KL explosion).
    """
    def __init__(self, latent_dim, context_dropout_rate=0):
        super().__init__()
        self.context_dropout_rate = context_dropout_rate

        # Input: Global (32) + Local (32) = 96
        input_dim = latent_dim + latent_dim

        # 1. Maintain Width:
        # Instead of compressing, we project up or keep equal.
        # 128 gives enough capacity to mix Global and Previous-Local info.
        hidden_dim = 2*input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, 2 * latent_dim, bias=False) # Outputs (mu, var_param)

        self.activation = nn.GELU()

        # --- CRITICAL: Near-Zero Initialization ---
        # Initialize output to be very close to N(0, 1) parameters.
        # mu -> 0, sigma_param -> 0 (which becomes softplus(0) ~ 0.69)
        # This matches the initialization of your Encoder.
        torch.nn.init.normal_(self.fc_out.weight, mean=0.0, std=0.001)
        # torch.nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, z_t, z_i_minus_1):
        """
        Args:
            z_t (Tensor): Global latent (B, D_global)
            z_i_minus_1 (Tensor): Previous local latent (B, D_local)
        """

        # --- 2. Context Dropout (The "Firewall") ---
        # Randomly zero out the previous sentence information.
        # This forces the network to look at z_t to guess the current sentence state.
        if self.training and self.context_dropout_rate > 0:
            mask_prob = torch.rand(z_t.shape[0], 1, device=z_t.device)
            # If random > rate, keep signal (1.0). Else drop (0.0).
            keep_mask = (mask_prob > self.context_dropout_rate).float()
            z_i_minus_1 = z_i_minus_1 * keep_mask

        # Concatenate inputs
        h = torch.cat([z_t, z_i_minus_1], dim=-1)

        # Block 1
        x = self.fc1(h)
        x = self.ln1(x)
        x = self.activation(x)

        # Block 2
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.activation(x)

        # Output Head
        output = self.fc_out(x)

        mu, raw_var_score = output.chunk(2, dim=-1)

        # Robust Softplus
        sigma2 = F.softplus(raw_var_score) + 1e-6

        return mu, sigma2
