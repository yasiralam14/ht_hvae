import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import torch
import numpy as np


def compute_active_units(mu_global, mu_local, threshold=0.01):
    """
    mu_global: (B, D)
    mu_local:  (B, S, D)
    Returns metrics dict (includes a wandb.Image heatmap).
    """
    assert mu_global.dim() == 2, f"mu_global should be (B,D), got {mu_global.shape}"
    assert mu_local.dim() == 3, f"mu_local should be (B,S,D), got {mu_local.shape}"
    B, S, D = mu_local.shape
    assert mu_global.shape == (B, D), f"Expected mu_global {(B,D)}, got {mu_global.shape}"

    # Detach so logging doesn't keep graphs around
    mu_g = mu_global.detach()
    mu_l = mu_local.detach()

    # 1) Global AU
    global_vars = mu_g.var(dim=0, unbiased=False)          # (D,)
    num_active_global = (global_vars > threshold).sum().item()

    # 2) Local AU (aggregate across all sentences)
    local_flat = mu_l.flatten(0, 1)                        # (B*S, D)
    local_vars_agg = local_flat.var(dim=0, unbiased=False) # (D,)
    num_active_local = (local_vars_agg > threshold).sum().item()

    # 3) Local activity map per sentence position
    local_activity_map = mu_l.var(dim=0, unbiased=False)   # (S, D)

    metrics = {
        "AU/Global_Count": num_active_global,
        "AU/Local_Count": num_active_local,
        "AU/Global_Variance_Avg": global_vars.mean().item(),
        "AU/Local_Variance_Avg": local_vars_agg.mean().item(),
    }

    # Optional: per-sentence active counts (often useful)
    per_sent_active = (local_activity_map > threshold).sum(dim=1)  # (S,)
    metrics["AU/Local_Count_PerSentence_Mean"] = per_sent_active.float().mean().item()

    # Visualization (binary mask)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap((local_activity_map.cpu().numpy() > threshold), ax=ax, cbar=False)
    ax.set_title("Active Units per Sentence Position (Binary)")
    ax.set_ylabel("Sentence Index")
    ax.set_xlabel("Latent Dimension")

    metrics["AU/Local_Heatmap"] = wandb.Image(fig)
    plt.close(fig)

    return metrics
