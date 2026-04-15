# function to test ability of latents to discriminate between other sentences, un restricted
def test_latent_discrimination(encoder, decoder, dataloader, device, num_negatives=5):
    """
    Discrimination accuracy:
    For each (B,S) sentence slot that is real, count a win if
    NLL(x_true | z_true) < NLL(x_neg | z_true)
    """
    encoder.eval()
    decoder.eval()

    total = 0
    correct = 0

    print(f"Running Discrimination Test (1 Positive vs {num_negatives} Negatives)...")

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                enc_input_ids = batch['enc_input_ids'].to(device)
                enc_word_mask = batch['enc_word_mask'].to(device)
                dec_input_ids = batch['dec_input_ids'].to(device)
                dec_word_mask = batch['dec_word_mask'].to(device)
            else:
                raise ValueError("Expected dict batch with enc/dec ids + masks")

            B, S, W = dec_input_ids.shape

            # Latents (use mu for a cleaner analysis if you want)
            mu_t, logvar_t, mu_i, logvar_i = encoder(dec_input_ids, dec_word_mask)
            z_t = reparameterize(mu_t, logvar_t)
            z_i = reparameterize(mu_i, logvar_i)

            # Positive
            pos_logits, _, _ = decoder(dec_input_ids, dec_word_mask, z_t, z_i)
            pos_nll, is_real = compute_sentence_nll(pos_logits, dec_input_ids, dec_word_mask)

            for _ in range(num_negatives):
                perm = torch.randperm(B, device=device)
                neg_ids  = dec_input_ids[perm]
                neg_mask = dec_word_mask[perm]

                neg_logits, _, _ = decoder(neg_ids, neg_mask, z_t, z_i)
                neg_nll, _ = compute_sentence_nll(neg_logits, neg_ids, neg_mask)

                # Compare only real sentences (from the POS sample slots)
                wins = (pos_nll < neg_nll) & is_real
                correct += wins.sum().item()
                total += is_real.sum().item()

    acc = correct / max(total, 1)
    print(f"Discrimination Accuracy: {acc:.2%}  (total comparisons: {total})")
    return acc

def compute_sentence_nll(logits, target_ids, mask, eps=1e-8):
    """
    Returns:
      sent_nll_avg: (B, S) average NLL per real token in the sentence
      sent_is_real: (B, S) bool mask: True if sentence has >=1 real token
    """
    B, S, W, V = logits.shape

    # Flatten
    flat_logits  = logits.reshape(-1, V)          # (B*S*W, V)
    flat_targets = target_ids.reshape(-1)         # (B*S*W,)
    flat_mask    = mask.reshape(-1).float()       # (B*S*W,)

    # Per-token CE
    per_token = F.cross_entropy(flat_logits, flat_targets, reduction='none')  # (B*S*W,)

    # Mask padding
    per_token = per_token * flat_mask

    # Back to (B, S, W)
    per_token = per_token.view(B, S, W)
    mask3     = mask.float().view(B, S, W)

    # Token counts per sentence
    tok_counts = mask3.sum(dim=2)                 # (B, S)
    sent_is_real = tok_counts > 0                 # (B, S)

    # Average NLL per token (avoid div0)
    sent_nll_sum = per_token.sum(dim=2)           # (B, S)
    sent_nll_avg = sent_nll_sum / (tok_counts + eps)

    return sent_nll_avg, sent_is_real