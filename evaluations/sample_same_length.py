def sample_length_matched_indices(lengths, valid_mask, tol=0.10):
    """
    lengths: (B,) token counts
    valid_mask: (B,) bool (must be True for selectable negatives)
    Returns neg_idx: (B,) where neg_idx[i] != i and valid_mask[neg_idx[i]] = True
    """
    device = lengths.device
    B = lengths.size(0)
    neg_idx = torch.empty(B, dtype=torch.long, device=device)

    all_idx = torch.arange(B, device=device)

    for i in range(B):
        if not valid_mask[i]:
            # doesn't matter; will be ignored by is_real anyway
            # pick something valid to avoid crash
            candidates = all_idx[valid_mask]
            neg_idx[i] = candidates[0] if len(candidates) else i
            continue

        li = lengths[i].float()
        # candidates: valid, not self, length within tol
        candidates = all_idx[
            valid_mask &
            (all_idx != i) &
            (torch.abs(lengths.float() - li) <= tol * torch.clamp(li, min=1.0))
        ]

        if len(candidates) == 0:
            # fallback: any other valid
            candidates = all_idx[valid_mask & (all_idx != i)]

        neg_idx[i] = candidates[torch.randint(0, len(candidates), (1,), device=device)] if len(candidates) else i

    return neg_idx