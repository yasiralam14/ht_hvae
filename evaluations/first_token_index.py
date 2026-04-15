def _first_content_index(ids, msk, bos_id=50256, skip_bos=True):
    """
    ids, msk: (N,W), msk bool
    Returns:
      first_idx: (N,) index of first "content" token
      has_token: (N,) bool
    """
    N, W = ids.shape
    pos = torch.arange(W, device=ids.device).unsqueeze(0).expand(N, W)

    valid = msk
    if skip_bos:
        valid = valid & (ids != bos_id)

    big = torch.full_like(pos, W)
    first_idx = torch.where(valid, pos, big).min(dim=1).values
    has_token = first_idx < W
    return first_idx, has_token
