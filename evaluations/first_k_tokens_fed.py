# waht if feed first k tokens of gold can the model predict the rest
@torch.no_grad()
def test_first_k_tokens_gold_history(
    model, encoder, dataloader, device,
    K=8, bos_id=50256, max_batches=None,
    use_mu=True, shuffle_global=False, skip_bos=True,
):
    model.eval(); encoder.eval()

    sum_nll_true = 0.0
    sum_nll_shuf = 0.0
    sum_acc_true = 0.0
    sum_acc_shuf = 0.0
    count = 0

    for b_idx, batch in enumerate(dataloader):
        if max_batches is not None and b_idx >= max_batches:
            break

        dec_input_ids = batch["dec_input_ids"].to(device)
        dec_word_mask = batch["dec_word_mask"].to(device)

        out = _compute_plans_flat(model, encoder, dec_input_ids, dec_word_mask,
                                  use_mu=use_mu, shuffle_global=shuffle_global)
        if out is None:
            continue
        ids, msk, plan_true, plan_shuf = out  # (N,W), (N,W), (N,D)
        N, W = ids.shape

        first_idx, has_tok = _first_content_index(ids, msk, bos_id=bos_id, skip_bos=skip_bos)
        if has_tok.sum().item() == 0:
            continue

        ids = ids[has_tok]
        msk = msk[has_tok]
        first_idx = first_idx[has_tok]
        plan_true = plan_true[has_tok]
        plan_shuf = plan_shuf[has_tok]



        # loop over next K tokens after first_idx
        for j in range(K):
            t = first_idx + j
            valid = t < W
            if valid.sum().item() == 0:
                break

            t_idx = t[valid]
            ids_v = ids[valid]
            msk_v = msk[valid]
            plan_true_v = plan_true[valid]
            plan_shuf_v = plan_shuf[valid]

            # token at position t must be real
            ok = msk_v[torch.arange(ids_v.size(0), device=device), t_idx]
            if ok.sum().item() == 0:
                continue

            ids_ok = ids_v[ok]
            t_ok = t_idx[ok]
            plan_true_ok = plan_true_v[ok]
            plan_shuf_ok = plan_shuf_v[ok]

            # history is gold tokens up to t (exclusive)
            # per-example variable t, so we do a small loop (K is small)
            for n in range(ids_ok.size(0)):
                tt = int(t_ok[n].item())
                if tt == 0:
                    # no history: prefix-only
                    logits_true = _logits_next_from_prefix_only(model, plan_true_ok[n:n+1],bos_id)
                    logits_shuf = _logits_next_from_prefix_only(model, plan_shuf_ok[n:n+1],bos_id)
                else:
                    logits_true = _logits_next_from_prefix_and_history(model, plan_true_ok[n:n+1], ids_ok[n:n+1, :tt])
                    logits_shuf = _logits_next_from_prefix_and_history(model, plan_shuf_ok[n:n+1], ids_ok[n:n+1, :tt])

                target = ids_ok[n, tt].view(1)

                nll_true = F.cross_entropy(logits_true, target, reduction="sum").item()
                nll_shuf = F.cross_entropy(logits_shuf, target, reduction="sum").item()

                sum_nll_true += nll_true
                sum_nll_shuf += nll_shuf

                sum_acc_true += float((logits_true.argmax(-1) == target).item())
                sum_acc_shuf += float((logits_shuf.argmax(-1) == target).item())
                count += 1

    eps = 1e-9
    return {
        "K": K,
        "count": count,
        "mean_nll_true": sum_nll_true / (count + eps),
        "mean_nll_shuf": sum_nll_shuf / (count + eps),
        "mean_delta_nll": (sum_nll_shuf - sum_nll_true) / (count + eps),
        "mean_acc_true": sum_acc_true / (count + eps),
        "mean_acc_shuf": sum_acc_shuf / (count + eps),
    }