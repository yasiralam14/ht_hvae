def plot_planning_metrics(stats):
    """
    Adapted to handle missing KL keys and 'count' vs 'counts' mismatch.
    """
    
    # 1. Handle Key Mismatches
    # Map 'count' to 'counts' if necessary
    if "counts" not in stats and "count" in stats:
        stats["counts"] = stats["count"]

    # 2. Determine valid range
    # Assumes stats are LISTS. If scalars, this len() check will fail or be 1.
    cutoff = len(stats["mean_delta_nll"])
    
    # X-axis: Token Steps
    steps = np.arange(1, cutoff + 1)
    
    # Helper to slice data (adjust slice depending on if you have padding at index 0)
    # If your lists start directly at t=1, use [0:cutoff]. If index 0 is padding, use [1:cutoff+1].
    # This version assumes input lists match 'steps' length exactly.
    def get_slice(key):
        data = stats.get(key, np.zeros(cutoff)) # Fallback to zeros if missing
        return np.array(data)

    # 3. Setup Figure
    # We hide Plot A (KL) if keys are missing
    has_kl = "mean_kl_true_shuf" in stats
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Latent Planning Horizon Diagnostics', fontsize=18, weight='bold')
    
    # --- PLOT A: Signal vs Noise (KL) ---
    ax_kl = axes[0, 0]
    if has_kl:
        kl_signal = get_slice("mean_kl_true_shuf")
        kl_noise  = get_slice("mean_kl_rand_rand")
        ax_kl.plot(steps, kl_signal, label='Signal', color='#1f77b4', marker='o')
        ax_kl.plot(steps, kl_noise, label='Noise', color='#7f7f7f', linestyle='--')
        ax_kl.legend()
    else:
        ax_kl.text(0.5, 0.5, "KL Data Missing", ha='center', fontsize=14, color='gray')
    ax_kl.set_title("1. Latent Signal (KL)")
    ax_kl.set_xlabel("Token Step")
    ax_kl.grid(True, alpha=0.3)

    # --- PLOT B: Latent Utility (Accuracy) ---
    ax_acc = axes[0, 1]
    acc_true = get_slice("mean_acc_true")
    acc_shuf = get_slice("mean_acc_shuf")
    
    ax_acc.plot(steps, acc_true, label='Acc w/ Plan', color='#2ca02c')
    ax_acc.plot(steps, acc_shuf, label='Acc w/o Plan', color='#ff7f0e', linestyle='--')
    ax_acc.fill_between(steps, acc_true, acc_shuf, color='#2ca02c', alpha=0.15)
    ax_acc.set_title("2. Prediction Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)
    
    # --- PLOT C: Helpfulness (Delta NLL) ---
    ax_delta = axes[1, 0]
    delta_nll = get_slice("mean_delta_nll")
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in delta_nll]
    ax_delta.bar(steps, delta_nll, color=colors, alpha=0.7)
    ax_delta.axhline(0, color='black', linewidth=1)
    ax_delta.set_title("3. Latent Helpfulness (Delta NLL)")
    ax_delta.grid(True, axis='y', alpha=0.3)

    # --- PLOT D: Reliability (Counts) ---
    ax_cnt = axes[1, 1]
    counts = get_slice("counts")
    ax_cnt.plot(steps, counts, color='black', marker='.')
    ax_cnt.set_title("4. Sample Validity (Count)")
    ax_cnt.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()