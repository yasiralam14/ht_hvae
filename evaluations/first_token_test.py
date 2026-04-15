@torch.no_grad()
def test_first_token_anchoring(
    model,
    encoder,
    dataloader,
    device,
    bos_id=50256,
    max_batches=None,
    use_mu=True,
    shuffle_global=False,
    skip_bos=True,
):
    """
    Predict the FIRST content token in each real sentence using ONLY the plan-prefix (no history).

    Returns:
      mean_nll_true, mean_nll_shuf, mean_delta_nll,
      mean_acc_true, mean_acc_shuf,
      count
    """
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
        ids, msk, plan_true, plan_shuf = out
        N, W = ids.shape

        # find first content token per sentence
        first_idx, has_tok = _first_content_index(ids, msk, bos_id=bos_id, skip_bos=skip_bos)
        if has_tok.sum().item() == 0:
            continue

        ids = ids[has_tok]
        first_idx = first_idx[has_tok]
        plan_true = plan_true[has_tok]
        plan_shuf = plan_shuf[has_tok]

        # targets: first content tokens
        targets = ids[torch.arange(ids.size(0), device=device), first_idx]  # (N,)


        logits_true = _logits_next_from_prefix_only(model, plan_true,bos_id)
        logits_shuf = _logits_next_from_prefix_only(model, plan_shuf,bos_id)

        nll_true = F.cross_entropy(logits_true, targets, reduction="none")
        nll_shuf = F.cross_entropy(logits_shuf, targets, reduction="none")

        sum_nll_true += float(nll_true.sum().item())
        sum_nll_shuf += float(nll_shuf.sum().item())

        pred_true = logits_true.argmax(dim=-1)
        pred_shuf = logits_shuf.argmax(dim=-1)
        sum_acc_true += float((pred_true == targets).float().sum().item())
        sum_acc_shuf += float((pred_shuf == targets).float().sum().item())

        count += int(targets.numel())

    eps = 1e-9
    mean_nll_true = sum_nll_true / (count + eps)
    mean_nll_shuf = sum_nll_shuf / (count + eps)

    return {
        "count": count,
        "mean_nll_true": mean_nll_true,
        "mean_nll_shuf": mean_nll_shuf,
        "mean_delta_nll": mean_nll_shuf - mean_nll_true,
        "mean_acc_true": sum_acc_true / (count + eps),
        "mean_acc_shuf": sum_acc_shuf / (count + eps),
    }

