@torch.no_grad()
def _embed_in_chunks(embed_fn, ids_2d, mask_2d, chunk=256):
    """
    embed_fn: (N,W), (N,W) -> (N,D)
    Returns embeddings on the same device as ids_2d.
    """
    device = ids_2d.device
    outs = []
    for start in range(0, ids_2d.size(0), chunk):
        end = min(start + chunk, ids_2d.size(0))
        emb = embed_fn(ids_2d[start:end], mask_2d[start:end])
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
        emb = emb.to(device)
        outs.append(emb)
    return torch.cat(outs, dim=0)


@torch.no_grad()
def latent_discrimination_sbert_closest_of_k(
    encoder,
    decoder,
    dataloader,
    device,
    compute_sentence_nll_avg,
    embed_fn,                  # from sbert_embedder_wrapper(...)
    reparameterize_fn=None,     # your reparameterize if use_mu=False
    use_mu=True,
    K=50,
    max_comparisons=10_000,
    anchor_chunk=32,
    pair_chunk=256,            # passed to _nll_from_prefix_pairs
    embed_chunk=256,           # SBERT batching
    seed=0,
):
    """
    For each anchor i (real sentence) in a batch:
      pos = NLL(sentence_i | plan_i)
      sample K negatives j from (B*S real sentences)
      pick closest negative by SBERT cosine similarity: argmax cos(sbert(i), sbert(j))
      neg = NLL(sentence_j | plan_i)
      win if pos < neg
      margin = neg - pos

    Stops after max_comparisons anchors across dataloader.
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
    chosen_neg_nlls = []
    chosen_cos_sims = []

    for batch in dataloader:
        if total >= max_comparisons:
            break

        dec_input_ids = batch["dec_input_ids"].to(device)   # (B,S,W)
        dec_word_mask = batch["dec_word_mask"].to(device)   # (B,S,W)

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

        # --- POS NLL (correct sentence + correct plan) ---
        pos_logits, _, _ = decoder(dec_input_ids, dec_word_mask, z_t, z_i)
        pos_nll, is_real = compute_sentence_nll_avg(pos_logits, dec_input_ids, dec_word_mask)  # (B,S)

        ids_flat = dec_input_ids.view(N, W)
        msk_flat = dec_word_mask.view(N, W)
        pos_nll_flat = pos_nll.view(N)
        is_real_flat = is_real.view(N)

        real_idx = torch.nonzero(is_real_flat, as_tuple=False).squeeze(1).cpu()  # global indices in [0..N)
        P = int(real_idx.numel())
        if P < 2:
            continue

        # --- prefixes for plans (plan_i -> prefix_i), for all sentences ---
        plans_flat = _build_prefix_embeds_from_latents(decoder, z_t, z_i)  # (N, prefix_len, H)

        # --- SBERT embeddings for ALL real sentences (once per batch) ---
        real_idx_dev = real_idx.to(device)
        real_ids = ids_flat[real_idx_dev]     # (P,W)
        real_msk = msk_flat[real_idx_dev]     # (P,W)
        real_emb = _embed_in_chunks(embed_fn, real_ids, real_msk, chunk=embed_chunk)  # (P,D)

        # normalize once for cosine similarity
        real_emb = F.normalize(real_emb, dim=-1)

        # mapping: global sentence index -> row in real_emb
        index_map = torch.full((N,), -1, dtype=torch.long, device="cpu")
        index_map[real_idx] = torch.arange(P, dtype=torch.long)

        # random anchor order
        perm = torch.randperm(P, generator=gen)
        anchors_all = real_idx[perm]  # CPU global indices

        ptr = 0
        while ptr < P and total < max_comparisons:
            A = min(anchor_chunk, P - ptr, max_comparisons - total)
            anchors = anchors_all[ptr:ptr + A]  # CPU global indices
            ptr += A

            # sample K negatives from real pool, excluding anchor itself
            picks = torch.randint(0, P, (A, K), generator=gen)   # CPU indices into [0..P)
            neg_rows = picks.clone()                              # rows in real_emb
            anchor_rows = index_map[anchors]                      # (A,) rows in real_emb

            # enforce row != anchor_row
            neq = (neg_rows != anchor_rows.view(-1, 1))
            while not bool(neq.all()):
                bad = torch.nonzero(~neq, as_tuple=False)
                neg_rows[bad[:, 0], bad[:, 1]] = torch.randint(0, P, (bad.size(0),), generator=gen)
                neq = (neg_rows != anchor_rows.view(-1, 1))

            # cosine similarity: choose closest (max cos)
            a_emb = real_emb[anchor_rows.to(device)]              # (A,D)
            n_emb = real_emb[neg_rows.to(device)]                 # (A,K,D)
            cos = (n_emb * a_emb.unsqueeze(1)).sum(dim=-1)        # (A,K)
            best_k = cos.argmax(dim=1)                            # (A,)

            # chosen negative global indices:
            # we sampled negatives by rows-in-real_emb; convert row->global via real_idx[row]
            chosen_neg_rows = neg_rows[torch.arange(A), best_k.cpu()]  # CPU rows
            chosen_neg_global = real_idx[chosen_neg_rows]              # CPU global idx in [0..N)

            # compute NLL(sentence_neg | plan_anchor)
            anchors_dev = anchors.to(device)
            chosen_neg_dev = chosen_neg_global.to(device)

            ids_pairs = ids_flat[chosen_neg_dev]     # (A,W)
            msk_pairs = msk_flat[chosen_neg_dev]     # (A,W)
            plan_pairs = plans_flat[anchors_dev]    # (A,prefix_len,H)

            neg_nll, _ = _nll_from_prefix_pairs(
                decoder,
                compute_sentence_nll_avg,
                ids_pairs,
                msk_pairs,
                plan_pairs,
                pair_chunk=pair_chunk,
            )  # (A,)

            pos = pos_nll_flat[anchors_dev]          # (A,)
            margin = (neg_nll - pos)                 # (A,)

            wins += int((margin > 0).sum().item())
            ties += int((margin == 0).sum().item())
            total += A

            margins.append(margin.detach().cpu())
            pos_nlls.append(pos.detach().cpu())
            chosen_neg_nlls.append(neg_nll.detach().cpu())
            chosen_cos_sims.append(cos.max(dim=1).values.detach().cpu())

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
    chosen_neg_nlls = torch.cat(chosen_neg_nlls)
    chosen_cos_sims = torch.cat(chosen_cos_sims)

    accuracy = wins / total
    q25, q50, q75 = torch.quantile(margins, torch.tensor([0.25, 0.50, 0.75])).tolist()

    return {
        "total": total,
        "wins": wins,
        "ties": ties,
        "accuracy": accuracy,
        "mean_pos_nll": float(pos_nlls.mean().item()),
        "mean_chosen_neg_nll": float(chosen_neg_nlls.mean().item()),
        "mean_margin": float(margins.mean().item()),
        "margin_p25": float(q25),
        "margin_median": float(q50),
        "margin_p75": float(q75),
        "mean_chosen_cos_sim": float(chosen_cos_sims.mean().item()),
    }