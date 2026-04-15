def oracle_prefix_sweep(encoder, decoder, dataloader, device,
                        K_list=(0,5,10,20,40,80),
                        max_new_tokens=200, num_batches=20,
                        use_mu=True, bos_id=50256, eos_id=50256):
    encoder.eval()
    decoder.eval()

    K_list = list(K_list)
    acc_sum = {K: 0.0 for K in K_list}
    acc_cnt = {K: 0   for K in K_list}

    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):
            if b_idx >= num_batches:
                break

            enc_input_ids = batch["enc_input_ids"].to(device)
            enc_word_mask = batch["enc_word_mask"].to(device)
            dec_input_ids = batch["dec_input_ids"].to(device)
            dec_word_mask = batch["dec_word_mask"].to(device)

            B, S, W = dec_input_ids.shape

            mu_t, sigma2_t, mu_i, sigma2_i = encoder(dec_input_ids, dec_word_mask)
            z_t = mu_t if use_mu else reparameterize(mu_t, sigma2_t)
            z_i = mu_i if use_mu else reparameterize(mu_i, sigma2_i)

            # Compute plan vectors like your inference
            z_t_exp = z_t.unsqueeze(1).repeat(1, S, 1)
            gru_in = torch.cat([z_i, z_t_exp], dim=-1)
            h0 = decoder.gru_initial_projection(z_t).unsqueeze(0).repeat(decoder.gru.num_layers, 1, 1)
            plan_vecs, _ = decoder.gru(gru_in, h0)  # (B,S,D_model)

            for i in range(B):
                for s in range(S):
                    # skip padding sentences
                    if dec_word_mask[i, s].sum().item() == 0:
                        continue

                    plan = plan_vecs[i, s].unsqueeze(0)  # (1,D)


                    gold_ids = dec_input_ids[i, s]   # (W,)
                    gold_mask = dec_word_mask[i, s]  # (W,)

                    for K in K_list:
                        a = greedy_decode_with_oracle_prefix(
                            decoder, plan, gold_ids, gold_mask,
                            K=K, max_new_tokens=max_new_tokens,
                            bos_id=bos_id, eos_id=eos_id
                        )
                        if a is None:
                            continue
                        acc_sum[K] += a
                        acc_cnt[K] += 1

    results = {K: (acc_sum[K] / max(acc_cnt[K], 1)) for K in K_list}
    print("Oracle prefix-length curve (avg token-acc on remaining tokens):")
    for K in K_list:
        print(f"  K={K:>3}: acc={results[K]:.4f}  (n={acc_cnt[K]})")
    return results