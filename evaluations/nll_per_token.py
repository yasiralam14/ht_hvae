# Sentence nll avergaed per token
def compute_sentence_nll_avg(logits, target_ids, mask, eps=1e-8):
    """
    logits: (B,S,W,V)
    target_ids: (B,S,W)
    mask: (B,S,W) 1 for real tokens, 0 for pad
    Returns:
      nll_avg: (B,S)  average NLL per real token
      is_real: (B,S)  True if sentence has >=1 real token
    """
    B, S, W, V = logits.shape
    flat_logits = logits.reshape(-1, V)
    flat_tgt = target_ids.reshape(-1)
    flat_mask = mask.reshape(-1).float()

    per_tok = F.cross_entropy(flat_logits, flat_tgt, reduction="none") * flat_mask
    per_tok = per_tok.view(B, S, W)
    mask3 = mask.float()

    tok_counts = mask3.sum(dim=2)                 # (B,S)
    is_real = tok_counts > 0
    nll_sum = per_tok.sum(dim=2)                  # (B,S)
    nll_avg = nll_sum / (tok_counts + eps)

    return nll_avg, is_real