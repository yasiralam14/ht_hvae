@torch.no_grad()
def noisy_history_plan_effect_curve(
    model,
    encoder,
    dataloader,
    device,
    max_steps=30,
    p_corrupt=0.2,
    corruption_mode="model_greedy",  # "model_greedy" | "random"
    target_mode="gold",              # "gold" | "noisy"
    bos_id=50256,
    use_mu=True,
    shuffle_global=False,
    min_count=256,
    max_batches=None,
):
    """
    Build a noisy version of each sentence by corrupting tokens as you roll forward.

    corruption_mode:
      - "model_greedy": when corrupting position j, replace with argmax token under TRUE plan given noisy history <j
      - "random": replace with uniform random token id (0..V-1)

    target_mode:
      - "gold": evaluate NLL/acc on the gold next token (can become inconsistent with noisy history)
      - "noisy": evaluate NLL/acc on the noisy next token (tests self-consistency on perturbed trajectory)

    Returns per-step curves (only trust up to cutoff_t where counts>=min_count).
    """
    model.eval(); encoder.eval()

    sum_delta = torch.zeros(max_steps, device=device)
    sum_nll_true = torch.zeros(max_steps, device=device)
    sum_nll_shuf = torch.zeros(max_steps, device=device)
    sum_acc_true = torch.zeros(max_steps, device=device)
    sum_acc_shuf = torch.zeros(max_steps, device=device)
    cnt = torch.zeros(max_steps, device=device)

    # vocab size for random corruption
    # (assumes final_linear outputs vocab logits)
    V = model.final_linear.out_features

    for b_idx, batch in enumerate(dataloader):
        if max_batches is not None and b_idx >= max_batches:
            break

        dec_input_ids = batch["dec_input_ids"].to(device)
        dec_word_mask = batch["dec_word_mask"].to(device)

        out = _compute_plans_flat(model, encoder, dec_input_ids, dec_word_mask,
                                  use_mu=use_mu, shuffle_global=shuffle_global)
        if out is None:
            continue
        ids, msk, plan_true, plan_shuf = out  # (N,W), (N,W), (N,Dm), (N,Dm)
        N, W = ids.shape
        T = min(max_steps, W)



        # ---- build noisy sequence (right-pad aware) ----
        noisy = ids.clone()

        # keep bos as-is if present; otherwise start at j=0
        start_j = 1 if (W > 0 and (noisy[:, 0] == bos_id).all().item()) else 0

        for j in range(start_j, T):
            valid_j = msk[:, j]  # token position j exists?
            if valid_j.sum().item() == 0:
                continue

            corrupt_mask = (torch.rand(N, device=device) < p_corrupt) & valid_j
            if corrupt_mask.sum().item() == 0:
                continue

            if corruption_mode == "random":
                noisy[corrupt_mask, j] = torch.randint(0, V, (int(corrupt_mask.sum().item()),), device=device)
            elif corruption_mode == "model_greedy":
                # predict token at position j given noisy history <j under TRUE plan
                if j == 0:
                    # no history; use prefix-only
                    logits_next = _logits_next_from_prefix_only(model, plan_true, bos_id)
                else:
                    logits_next = _logits_next_from_prefix_and_history(model, plan_true, noisy[:, :j])
                pred = logits_next.argmax(dim=-1)
                noisy[corrupt_mask, j] = pred[corrupt_mask]
            else:
                raise ValueError(f"Unknown corruption_mode: {corruption_mode}")

        # ---- evaluate per-step next-token prediction under noisy history ----
        for t in range(1, T):
            valid_t = msk[:, t]
            n_valid = int(valid_t.sum().item())
            if n_valid == 0:
                continue

            hist = noisy[:, :t]  # (N,t)

            logits_true = _logits_next_from_prefix_and_history(model, plan_true, hist)
            logits_shuf = _logits_next_from_prefix_and_history(model, plan_shuf, hist)

            targets = ids[:, t] if target_mode == "gold" else noisy[:, t]

            nll_true_vec = F.cross_entropy(logits_true[valid_t], targets[valid_t], reduction="none")
            nll_shuf_vec = F.cross_entropy(logits_shuf[valid_t], targets[valid_t], reduction="none")

            sum_nll_true[t] += nll_true_vec.sum()
            sum_nll_shuf[t] += nll_shuf_vec.sum()
            sum_delta[t] += (nll_shuf_vec - nll_true_vec).sum()

            pred_true = logits_true.argmax(dim=-1)
            pred_shuf = logits_shuf.argmax(dim=-1)
            sum_acc_true[t] += (pred_true[valid_t] == targets[valid_t]).float().sum()
            sum_acc_shuf[t] += (pred_shuf[valid_t] == targets[valid_t]).float().sum()

            cnt[t] += n_valid

    eps = 1e-9
    counts = cnt.detach().cpu().long().tolist()

    mean_delta = (sum_delta / (cnt + eps)).detach().cpu().tolist()
    mean_nll_true = (sum_nll_true / (cnt + eps)).detach().cpu().tolist()
    mean_nll_shuf = (sum_nll_shuf / (cnt + eps)).detach().cpu().tolist()
    mean_acc_true = (sum_acc_true / (cnt + eps)).detach().cpu().tolist()
    mean_acc_shuf = (sum_acc_shuf / (cnt + eps)).detach().cpu().tolist()

    cutoff = 0
    for t in range(1, max_steps):
        if counts[t] >= min_count:
            cutoff = t

    return {
        "p_corrupt": p_corrupt,
        "corruption_mode": corruption_mode,
        "target_mode": target_mode,
        "mean_delta_nll": mean_delta,
        "mean_nll_true": mean_nll_true,
        "mean_nll_shuf": mean_nll_shuf,
        "mean_acc_true": mean_acc_true,
        "mean_acc_shuf": mean_acc_shuf,
        "counts": counts,
        "cutoff_t": cutoff,
    }
