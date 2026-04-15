import torch
import torch.nn as nn

class HT_HVAE_Loss(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.vocab_size = hyperparameters["vocab_size"]
        self.pad_idx = hyperparameters["pad_index"]
        self.latent_dim = hyperparameters["latent_dim"]
        self.local_latent = self.latent_dim

        self.cross_entropy_loss = nn.CrossEntropyLoss(
            ignore_index=self.pad_idx, reduction="none"
        )

        # free-bits thresholds (nats per dim)
        self.free_bits_global = 0.5
        self.free_bits_local = 0.5

    def forward(
        self,
        mu_t_q, sigma2_t_q,
        mu_i_q, sigma2_i_q,
        reconstruction_logits,
        mu_i_p, sigma2_i_p,
        target_ids, word_mask,
        local_kl_beta=0.5, global_kl_beta=0.1,
    ):
        eps = 1e-9
        B, S, W = target_ids.shape

        # -------------------------
        # A) Reconstruction: mean over batch (sum tokens per example, then avg over batch)
        # -------------------------
        shift_logits = reconstruction_logits[..., :-1, :].contiguous()  # (B,S,W-1,V)
        shift_labels = target_ids[..., 1:].contiguous()                 # (B,S,W-1)

        nll_flat = self.cross_entropy_loss(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1)
        )  # (B*S*(W-1),) with PAD ignored as 0

        nll = nll_flat.view(B, S, W - 1)
        token_mask = (shift_labels != self.pad_idx).float()             # (B,S,W-1)

        recon_per_ex = (nll * token_mask).sum(dim=(1, 2))               # (B,)
        reconstruction_loss = recon_per_ex.mean()                       # scalar (avg over batch)

        num_active_tokens = token_mask.sum().float()
        mean_token_loss = recon_per_ex.sum() / (num_active_tokens + eps)  # scalar (avg per token)

        # Sentence mask: 1 for real sentences, 0 for padded sentences
        sentence_mask = (word_mask.sum(dim=-1) > 0).float()             # (B,S)

        # -------------------------
        # B) Global KL: sum dims per example, mean over batch
        # -------------------------
        global_kl_raw = 0.5 * (
            sigma2_t_q + mu_t_q.pow(2) - 1.0 - torch.log(sigma2_t_q + eps)
        )  # (B,D)

        global_frac_dims_clamped = (global_kl_raw < self.free_bits_global).float().mean()

        global_kl_charged = torch.clamp(global_kl_raw, min=self.free_bits_global)  # (B,D)

        global_kl_raw_per_ex = global_kl_raw.sum(dim=-1)              # (B,)
        global_kl_charged_per_ex = global_kl_charged.sum(dim=-1)      # (B,)

        global_kl_raw_mean = global_kl_raw_per_ex.mean()              # scalar
        global_kl_loss = global_kl_charged_per_ex.mean()              # scalar

        # -------------------------
        # C) Local KL: sum dims, sum active sentences per example, mean over batch
        # -------------------------
        mu_p_flat = mu_i_p.reshape(-1, self.local_latent)
        sigma2_p_flat = sigma2_i_p.reshape(-1, self.local_latent)
        mu_q_flat = mu_i_q.reshape(-1, self.local_latent)
        sigma2_q_flat = sigma2_i_q.reshape(-1, self.local_latent)

        term1 = torch.log(sigma2_p_flat + eps) - torch.log(sigma2_q_flat + eps)
        term2 = (sigma2_q_flat + (mu_q_flat - mu_p_flat).pow(2)) / (sigma2_p_flat + eps)
        local_kl_raw_flat = 0.5 * (term1 + term2 - 1.0)               # (B*S,D)

        local_kl_raw = local_kl_raw_flat.view(B, S, self.local_latent)  # (B,S,D)

        active_mask_bsd = sentence_mask.unsqueeze(-1)                   # (B,S,1)
        local_under = (local_kl_raw < self.free_bits_local).float() * active_mask_bsd
        local_frac_dims_clamped = local_under.sum() / (active_mask_bsd.sum() * self.local_latent + eps)

        local_kl_charged = torch.clamp(local_kl_raw, min=self.free_bits_local)  # (B,S,D)

        local_kl_raw_per_sent = local_kl_raw.sum(dim=-1)               # (B,S)
        local_kl_charged_per_sent = local_kl_charged.sum(dim=-1)       # (B,S)

        local_kl_raw_per_ex = (local_kl_raw_per_sent * sentence_mask).sum(dim=-1)        # (B,)
        local_kl_charged_per_ex = (local_kl_charged_per_sent * sentence_mask).sum(dim=-1) # (B,)

        local_kl_raw_mean = local_kl_raw_per_ex.mean()                 # scalar
        local_kl_loss = local_kl_charged_per_ex.mean()                 # scalar

        # -------------------------
        # D) Total
        # -------------------------
        total_loss = (
            reconstruction_loss
            + global_kl_beta * global_kl_loss
            + local_kl_beta * local_kl_loss
        )

        total_kl_unweighted = global_kl_loss + local_kl_loss
        kl_ratio = (reconstruction_loss / (total_kl_unweighted + 1e-8)).detach()

        return (
            total_loss,
            reconstruction_loss,   # avg over batch (sum tokens per ex, then mean)
            global_kl_loss,        # sum dims per ex, mean over batch (charged)
            local_kl_loss,         # sum dims+sentences per ex, mean over batch (charged)
            kl_ratio,
            mean_token_loss,       # avg per token (for logging)
            global_kl_raw_mean,    # sum dims per ex, mean over batch (raw)
            local_kl_raw_mean,     # sum dims+sentences per ex, mean over batch (raw)
            global_frac_dims_clamped,
            local_frac_dims_clamped,
        )
