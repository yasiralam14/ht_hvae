@torch.no_grad()
def latent_discrimination_hardest_of_k(
    encoder,
    decoder,
    dataloader,
    device,
    compute_sentence_nll_avg,
    reparameterize_fn=None,   # pass your reparameterize if use_mu=False
    use_mu=True,
    K=50,
    max_comparisons=10_000,
    anchor_chunk=16,          # anchors processed together
    pair_chunk=256,           # pairs per GPT forward (memory knob)
    seed=0,
):
    """
    For each anchor i in a batch:
      pos = NLL(sentence_i | plan_i)
      negs: sample K other sentences j, compute NLL(sentence_j | plan_i)
      hardest_neg = min_j neg
      win if pos < hardest_neg
      margin = hardest_neg - pos

    Stops after max_comparisons anchors across batches.
    """
    encoder.eval()
    decoder.eval()

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    total = 0
    wins = 0
    ties = 0

    margins = []
    pos_nlls = []
    hard_neg_nlls = []

    for batch in dataloader:
        if total >= max_comparisons:
            break

        # --- fetch inputs (adjust keys if your batch dict differs) ---
        dec_input_ids = batch["dec_input_ids"].to(device)     # (B,S,W)
        dec_word_mask = batch["dec_word_mask"].to(device)     # (B,S,W)

        B, S, W = dec_input_ids.shape
        N = B * S

        # --- encode latents ---
        mu_t, sigma2_t, mu_i, sigma2_i = encoder(dec_input_ids, dec_word_mask)
        if use_mu:
            z_t = mu_t
            z_i = mu_i
        else:
            assert reparameterize_fn is not None, "Pass reparameterize_fn if use_mu=False"
            z_t = reparameterize_fn(mu_t, sigma2_t)
            z_i = reparameterize_fn(mu_i, sigma2_i)

        # --- POS NLL: run your decoder normally (correct sentence + correct plan) ---
        pos_logits, _, _ = decoder(dec_input_ids, dec_word_mask, z_t, z_i)
        pos_nll, is_real = compute_sentence_nll_avg(pos_logits, dec_input_ids, dec_word_mask)  # (B,S), (B,S)

        # flatten
        ids_flat = dec_input_ids.view(N, W)
        msk_flat = dec_word_mask.view(N, W)
        pos_nll_flat = pos_nll.view(N)
        is_real_flat = is_real.view(N)

        real_idx = torch.nonzero(is_real_flat, as_tuple=False).squeeze(1).cpu()
        P = int(real_idx.numel())
        if P < 2:
            continue

        # --- build all prefixes for this batch plans (plan_i -> prefix_i) ---
        plans_flat = _build_prefix_embeds_from_latents(decoder, z_t, z_i)  # (N, prefix_len, H)

        # randomize anchor order
        perm = torch.randperm(P, generator=gen)
        anchors_all = real_idx[perm]  # CPU indices into [0..N)

        # process anchors in chunks
        ptr = 0
        while ptr < P and total < max_comparisons:
            A = min(anchor_chunk, P - ptr, max_comparisons - total)
            anchors = anchors_all[ptr:ptr + A]  # (A,) on CPU
            ptr += A

            anchors_dev = anchors.to(device)

            # sample K negatives (with replacement), excluding anchor itself
            pool = real_idx  # CPU
            pool_dev = pool.to(device)
            Ppool = pool.numel()

            # start with random picks in pool
            picks = torch.randint(0, Ppool, (A, K), generator=gen)  # CPU
            neg_idx = pool[picks]                                    # CPU indices into [0..N)
            # enforce neg != anchor (resample offending positions)
            neq = (neg_idx != anchors.view(-1, 1))
            while not bool(neq.all()):
                bad = torch.nonzero(~neq, as_tuple=False)
                newp = torch.randint(0, Ppool, (bad.size(0),), generator=gen)
                neg_idx[bad[:, 0], bad[:, 1]] = pool[newp]
                neq = (neg_idx != anchors.view(-1, 1))

            neg_idx_dev = neg_idx.to(device)  # (A,K)

            # build (A*K) PAIRS: (sentence_neg, prefix_anchor)
            ids_pairs = ids_flat[neg_idx_dev.reshape(-1)]  # (A*K, W)
            msk_pairs = msk_flat[neg_idx_dev.reshape(-1)]  # (A*K, W)

            plans_anchor = plans_flat[anchors_dev]                     # (A, P, H)
            plan_pairs = plans_anchor.unsqueeze(1).expand(A, K, -1, -1) # (A,K,P,H)
            plan_pairs = plan_pairs.reshape(A * K, plans_anchor.size(1), plans_anchor.size(2))

            neg_nll_pairs, _ = _nll_from_prefix_pairs(
                decoder,
                compute_sentence_nll_avg,
                ids_pairs,
                msk_pairs,
                plan_pairs,
                pair_chunk=pair_chunk,
            )
            neg_nll = neg_nll_pairs.view(A, K)               # (A,K)
            hard_neg = neg_nll.min(dim=1).values             # (A,)

            pos = pos_nll_flat[anchors_dev]                  # (A,)
            margin = (hard_neg - pos)                        # (A,)

            wins += int((margin > 0).sum().item())
            ties += int((margin == 0).sum().item())
            total += A

            margins.append(margin.detach().cpu())
            pos_nlls.append(pos.detach().cpu())
            hard_neg_nlls.append(hard_neg.detach().cpu())

    if total == 0:
        return {
            "total": 0,
            "accuracy": float("nan"),
            "wins": 0,
            "ties": 0,
            "margin_p25": float("nan"),
            "margin_median": float("nan"),
            "margin_p75": float("nan"),
        }

    margins = torch.cat(margins)
    pos_nlls = torch.cat(pos_nlls)
    hard_neg_nlls = torch.cat(hard_neg_nlls)

    accuracy = wins / total
    q25, q50, q75 = torch.quantile(margins, torch.tensor([0.25, 0.50, 0.75])).tolist()

    return {
        "total": total,
        "wins": wins,
        "ties": ties,
        "accuracy": accuracy,
        "mean_pos_nll": float(pos_nlls.mean().item()),
        "mean_hard_neg_nll": float(hard_neg_nlls.mean().item()),
        "mean_margin": float(margins.mean().item()),
        "margin_p25": float(q25),
        "margin_median": float(q50),
        "margin_p75": float(q75),
    }