def test_latent_discrimination_hardneg(encoder, decoder, dataloader, device,
                                       num_negatives=5, tol=0.10, use_mu=True):
    encoder.eval()
    decoder.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for batch in dataloader:
            enc_input_ids = batch["enc_input_ids"].to(device)
            enc_word_mask = batch["enc_word_mask"].to(device)
            dec_input_ids = batch["dec_input_ids"].to(device)
            dec_word_mask = batch["dec_word_mask"].to(device)

            B, S, W = dec_input_ids.shape

            mu_t, sigma2_t, mu_i, sigma2_i = encoder(dec_input_ids, dec_word_mask)
            z_t = mu_t if use_mu else reparameterize(mu_t, sigma2_t)
            z_i = mu_i if use_mu else reparameterize(mu_i, sigma2_i)

            pos_logits, _, _ = decoder(dec_input_ids, dec_word_mask, z_t, z_i)
            pos_nll, is_real = compute_sentence_nll_avg(pos_logits, dec_input_ids, dec_word_mask)

            lengths = dec_word_mask.sum(dim=2)  # (B,S)

            for _ in range(num_negatives):
                neg_ids = dec_input_ids.clone()
                neg_mask = dec_word_mask.clone()

                # For each sentence slot s, length-match within that slot
                for s in range(S):
                    lens_s = lengths[:, s]
                    valid_s = lens_s > 0
                    idx_s = sample_length_matched_indices(lens_s, valid_s, tol=tol)
                    neg_ids[:, s, :] = dec_input_ids[idx_s, s, :]
                    neg_mask[:, s, :] = dec_word_mask[idx_s, s, :]

                neg_logits, _, _ = decoder(neg_ids, neg_mask, z_t, z_i)
                neg_nll, _ = compute_sentence_nll_avg(neg_logits, neg_ids, neg_mask)

                wins = (pos_nll < neg_nll) & is_real
                correct += wins.sum().item()
                total += is_real.sum().item()

    acc = correct / max(total, 1)
    print(f"HardNeg Discrimination Acc: {acc:.2%} (comparisons: {total})")
    return acc