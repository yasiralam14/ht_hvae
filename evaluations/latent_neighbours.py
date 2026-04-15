def test_latent_neighbors(encoder, dataloader, tokenizer, device,
                          num_examples=5, use_mu_for_analysis=True):
    """
    Finds nearest neighbors in local latent space (z_i) for REAL (non-padding) sentences.
    Uses cosine distance.
    """
    encoder.eval()

    all_latents = []
    all_texts = []

    with torch.no_grad():
        for batch in dataloader:
            if not isinstance(batch, dict):
                raise ValueError("Expected dict batch with enc/dec ids + masks.")

            enc_input_ids = batch['enc_input_ids'].to(device)
            enc_word_mask = batch['enc_word_mask'].to(device)

            # Prefer decoder-side ids/mask for filtering/decoding if you have them
            dec_input_ids = batch.get('dec_input_ids', enc_input_ids).to(device)
            dec_word_mask = batch.get('dec_word_mask', enc_word_mask).to(device)

            mu_t, sigma2_t, mu_i, sigma2_i = encoder(dec_input_ids, dec_word_mask)

            # For analysis, mu is often cleaner than samples
            z_i = mu_i if use_mu_for_analysis else reparameterize(mu_i, sigma2_i)

            B, S, D = z_i.shape
            _, _, W = dec_input_ids.shape

            flat_latents = z_i.reshape(B * S, D)          # (B*S, D)
            flat_ids     = dec_input_ids.reshape(B * S, W)

            # Real sentence mask from decoder mask (more consistent with decoded ids)
            flat_m = dec_word_mask.reshape(B * S, W)
            is_real = flat_m.sum(dim=1) > 0               # (B*S,)

            # Keep only real sentences
            flat_latents = flat_latents[is_real]
            flat_ids     = flat_ids[is_real]

            # Now build aligned (latent, text) pairs
            for latent_vec, seq in zip(flat_latents, flat_ids):
                text = tokenizer.decode(seq.tolist(), skip_special_tokens=True).strip()
                if not text:
                    continue
                all_latents.append(latent_vec.detach().cpu().numpy())
                all_texts.append(text)

    if len(all_latents) < 10:
        print(f"Too few valid samples: {len(all_latents)}")
        return None

    all_latents = np.stack(all_latents, axis=0)

    all_latents = all_latents / (np.linalg.norm(all_latents, axis=1, keepdims=True) + 1e-8)


    print(f"Fitting NN on {len(all_latents)} valid samples...")
    nbrs = NearestNeighbors(n_neighbors=4, metric='cosine').fit(all_latents)

    num_examples = min(num_examples, len(all_latents))
    indices = np.random.choice(len(all_latents), num_examples, replace=False)

    for idx in indices:
        query_text = all_texts[idx]
        query_latent = all_latents[idx].reshape(1, -1)

        distances, neighbor_indices = nbrs.kneighbors(query_latent)

        print(f"\nQuery: {query_text}")
        print("-" * 50)

        found = 0
        for dist, nidx in zip(distances[0], neighbor_indices[0]):
            if nidx == idx:
                continue
            print(f"Neighbor {found+1} (Dist: {dist:.4f}): {all_texts[nidx]}")
            found += 1
            if found == 3:
                break

    return nbrs