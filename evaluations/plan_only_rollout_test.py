@torch.no_grad()
def plan_only_rollout_anchoring_curve(
    model,
    encoder,
    dataloader,
    device,
    K=20,
    bos_id=50256,
    use_mu=True,
    shuffle_global=False,
    skip_bos=True,
    max_batches=None,
    greedy=True,          # if False, sample
    temperature=1.0,
    top_k=None,
    min_count=256,
):
    """
    HARD test: prefix-only rollout.
    For each sentence:
      - start from prefix only (no gold history)
      - generate tokens step by step (ŷ_0, ŷ_1, ...)
      - at each step j, evaluate how well it predicts the GOLD token x_j
        *given its own generated history*.

    Computes per-step (j=0..K-1):
      - NLL_true / NLL_shuf on gold token
      - ΔNLL = NLL_shuf - NLL_true
      - acc_next_true / acc_next_shuf: argmax matches gold token (next-token under that history)
      - gen_match_true / gen_match_shuf: generated token equals gold token (actual rollout match)
      - counts[j]

    Notes:
      - Uses first content token index per sentence (skips BOS if skip_bos=True).
      - Ignores padding via masks and stops contributing when gold token doesn't exist.
    """
    model.eval(); encoder.eval()

    # running sums per step
    sum_nll_true = torch.zeros(K, device=device)
    sum_nll_shuf = torch.zeros(K, device=device)
    sum_delta    = torch.zeros(K, device=device)

    sum_acc_next_true = torch.zeros(K, device=device)
    sum_acc_next_shuf = torch.zeros(K, device=device)

    sum_gen_match_true = torch.zeros(K, device=device)
    sum_gen_match_shuf = torch.zeros(K, device=device)

    cnt = torch.zeros(K, device=device)

    # vocab size
    V = model.final_linear.out_features

    def sample_from_logits(logits):
        # logits: (N,V)
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)

        if top_k is not None and top_k > 0:
            topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
            probs = F.softmax(topk_vals, dim=-1)
            choice = torch.multinomial(probs, num_samples=1).squeeze(-1)
            return topk_idx[torch.arange(logits.size(0), device=logits.device), choice]
        else:
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

    for b_idx, batch in enumerate(dataloader):
        if max_batches is not None and b_idx >= max_batches:
            break

        dec_input_ids = batch["dec_input_ids"].to(device)   # (B,S,W)
        dec_word_mask = batch["dec_word_mask"].to(device)   # (B,S,W)

        out = _compute_plans_flat(
            model, encoder, dec_input_ids, dec_word_mask,
            use_mu=use_mu, shuffle_global=shuffle_global
        )
        if out is None:
            continue

        ids, msk, plan_true, plan_shuf = out   # ids/msk: (N,W)
        N, W = ids.shape

        # First content token index (per sentence)
        first_idx, has_tok = _first_content_index(ids, msk, bos_id=bos_id, skip_bos=skip_bos)
        if has_tok.sum().item() == 0:
            continue

        ids = ids[has_tok]
        msk = msk[has_tok]
        first_idx = first_idx[has_tok]
        plan_true = plan_true[has_tok]
        plan_shuf = plan_shuf[has_tok]

        N = ids.size(0)

        # Prefix embeddings


        # We'll store generated histories here (N,K)
        # Values for inactive rows don't matter because we'll mask them out via "active".
        gen_hist_true = torch.full((N, K), bos_id, device=device, dtype=ids.dtype)
        gen_hist_shuf = torch.full((N, K), bos_id, device=device, dtype=ids.dtype)

        for j in range(K):
            gold_pos = first_idx + j  # (N,)

            # which examples still have a real gold token at this position?
            in_bounds = gold_pos < W
            if in_bounds.sum().item() == 0:
                break

            idx_in = torch.nonzero(in_bounds, as_tuple=False).squeeze(-1)
            gold_pos_in = gold_pos[idx_in]
            msk_in = msk[idx_in]

            # real token mask at that gold position
            real = msk_in[torch.arange(idx_in.numel(), device=device), gold_pos_in]
            if real.sum().item() == 0:
                continue

            idx = idx_in[real]
            gold_pos_ok = gold_pos_in[real]

            # gold target token
            gold_tok = ids[idx, gold_pos_ok]  # (M,)

            # Build history = generated tokens so far (length j)
            if j == 0:
                logits_true = _logits_next_from_prefix_only(model, plan_true[idx],bos_id)
                logits_shuf = _logits_next_from_prefix_only(model, plan_shuf[idx],bos_id)
            else:
                hist_true = gen_hist_true[idx, :j]
                hist_shuf = gen_hist_shuf[idx, :j]
                logits_true = _logits_next_from_prefix_and_history(model, plan_true[idx], hist_true)
                logits_shuf = _logits_next_from_prefix_and_history(model, plan_shuf[idx], hist_shuf)

            # NLL on GOLD token under each condition
            nll_t = F.cross_entropy(logits_true, gold_tok, reduction="none")
            nll_s = F.cross_entropy(logits_shuf, gold_tok, reduction="none")

            sum_nll_true[j] += nll_t.sum()
            sum_nll_shuf[j] += nll_s.sum()
            sum_delta[j]    += (nll_s - nll_t).sum()

            # next-token accuracy (argmax == gold token)
            pred_next_true = logits_true.argmax(dim=-1)
            pred_next_shuf = logits_shuf.argmax(dim=-1)
            sum_acc_next_true[j] += (pred_next_true == gold_tok).float().sum()
            sum_acc_next_shuf[j] += (pred_next_shuf == gold_tok).float().sum()

            # choose generated token for rollout (greedy or sample)
            if greedy:
                gen_tok_true = pred_next_true
                gen_tok_shuf = pred_next_shuf
            else:
                gen_tok_true = sample_from_logits(logits_true)
                gen_tok_shuf = sample_from_logits(logits_shuf)

            # rollout-match vs gold (actual generated token equals gold)
            sum_gen_match_true[j] += (gen_tok_true == gold_tok).float().sum()
            sum_gen_match_shuf[j] += (gen_tok_shuf == gold_tok).float().sum()

            # write generated token into history buffer
            gen_hist_true[idx, j] = gen_tok_true
            gen_hist_shuf[idx, j] = gen_tok_shuf

            cnt[j] += gold_tok.numel()

    eps = 1e-9
    counts = cnt.detach().cpu().long().tolist()

    mean_delta = (sum_delta / (cnt + eps)).detach().cpu().tolist()
    mean_nll_t = (sum_nll_true / (cnt + eps)).detach().cpu().tolist()
    mean_nll_s = (sum_nll_shuf / (cnt + eps)).detach().cpu().tolist()

    mean_acc_next_t = (sum_acc_next_true / (cnt + eps)).detach().cpu().tolist()
    mean_acc_next_s = (sum_acc_next_shuf / (cnt + eps)).detach().cpu().tolist()

    mean_gen_match_t = (sum_gen_match_true / (cnt + eps)).detach().cpu().tolist()
    mean_gen_match_s = (sum_gen_match_shuf / (cnt + eps)).detach().cpu().tolist()

    cutoff = 0
    for j in range(K):
        if counts[j] >= min_count:
            cutoff = j

    return {
        "K": K,
        "greedy": greedy,
        "counts": counts,
        "cutoff_j": cutoff,
        "mean_delta_nll": mean_delta,
        "mean_nll_true": mean_nll_t,
        "mean_nll_shuf": mean_nll_s,
        "mean_acc_next_true": mean_acc_next_t,
        "mean_acc_next_shuf": mean_acc_next_s,
        "mean_gen_match_true": mean_gen_match_t,
        "mean_gen_match_shuf": mean_gen_match_s,
    }

