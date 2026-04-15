# function to see how the influence of latents on predictions for all batches in loader GOLD IDS IN HISTORY
def _per_example_kl(p_true, logp_true, logp_other):
    # p_true, logp_true, logp_other: (N, V)
    return (p_true * (logp_true - logp_other)).sum(dim=-1)

@torch.no_grad()
def plan_effect_curve_all_sentences(
    model,
    encoder,
    dataloader,
    device,
    max_steps=30,
    use_mu=True,
    shuffle_global=False,
    min_count=256,
    num_batches=None,   # None = all
):
    """
    Aggregates plan effect over a dataloader, using ALL real sentences in each batch.

    For each token step t (predict token t given tokens <t):
      ΔNLL(t) = NLL_shuf - NLL_true (positive => correct plan helps)
      acc_true(t), acc_shuf(t): next-token accuracy (argmax)
      KL_true_shuf(t) and KL_rand_rand(t) as controls

    Returns:
      dict with per-step mean curves + valid counts + suggested cutoff.
    """
    model.eval()
    encoder.eval()

    # Running sums (weighted by number of valid examples at that step)
    sum_delta = torch.zeros(max_steps, device=device)
    sum_nll_true = torch.zeros(max_steps, device=device)
    sum_nll_shuf = torch.zeros(max_steps, device=device)
    sum_kl_ts = torch.zeros(max_steps, device=device)
    sum_kl_rr = torch.zeros(max_steps, device=device)

    sum_acc_true = torch.zeros(max_steps, device=device)
    sum_acc_shuf = torch.zeros(max_steps, device=device)

    cnt = torch.zeros(max_steps, device=device)

    for b_idx, batch in enumerate(dataloader):
        if num_batches is not None and b_idx >= num_batches:
            break

        dec_input_ids = batch["dec_input_ids"].to(device)   # (B,S,W)
        dec_word_mask = batch["dec_word_mask"].to(device)   # (B,S,W)

        B, S, W = dec_input_ids.shape
        T = min(max_steps, W)

        # ----- Encode latents -----
        mu_t, logvar_t, mu_i, logvar_i = encoder(dec_input_ids, dec_word_mask)
        if use_mu:
            z_t, z_i = mu_t, mu_i
        else:
            z_t = mu_t + torch.randn_like(mu_t) * torch.exp(0.5 * logvar_t)
            z_i = mu_i + torch.randn_like(mu_i) * torch.exp(0.5 * logvar_i)

        # ----- Build plans (true) -----
        z_t_exp = z_t.unsqueeze(1).repeat(1, S, 1)                 # (B,S,Dz)
        gru_in_true = torch.cat([z_i, z_t_exp], dim=-1)            # (B,S,2Dz)
        h0 = model.gru_initial_projection(z_t).unsqueeze(0).repeat(model.gru.num_layers, 1, 1)
        plans_true, _ = model.gru(gru_in_true, h0)                 # (B,S,D_model)

        # ----- Build plans (shuf) -----
        perm = torch.randperm(B, device=device)
        z_i_shuf = z_i[perm]
        if shuffle_global:
            z_t_shuf = z_t[perm]
            z_t_exp_shuf = z_t_shuf.unsqueeze(1).repeat(1, S, 1)
        else:
            z_t_exp_shuf = z_t_exp
        plans_shuf, _ = model.gru(torch.cat([z_i_shuf, z_t_exp_shuf], dim=-1), h0)

        # ----- Random-vs-random control plans -----
        perm1 = torch.randperm(B, device=device)
        perm2 = torch.randperm(B, device=device)
        plans_r1, _ = model.gru(torch.cat([z_i[perm1], z_t_exp], dim=-1), h0)
        plans_r2, _ = model.gru(torch.cat([z_i[perm2], z_t_exp], dim=-1), h0)

        # ----- Flatten sentences: treat each sentence as an example -----
        ids = dec_input_ids.reshape(B * S, W)
        msk = dec_word_mask.reshape(B * S, W).bool()

        real_sent = msk.sum(dim=1) > 0
        if real_sent.sum().item() == 0:
            continue

        ids = ids[real_sent]
        msk = msk[real_sent]

        plans_true_f = plans_true.reshape(B * S, -1)[real_sent]
        plans_shuf_f = plans_shuf.reshape(B * S, -1)[real_sent]
        plans_r1_f   = plans_r1.reshape(B * S, -1)[real_sent]
        plans_r2_f   = plans_r2.reshape(B * S, -1)[real_sent]

        # Prefixes: (N, P, H)
        # helper to get logits
        def logits_from_prefix(curr_plan, hist_ids):
            # 1. Prepare Plan (Norm & Projections)
            norm_plan = model.plan_ln(curr_plan)
            
            # A. Calculate Embedding Bias (Gate * Plan)
            plan_emb = model.plan_to_emb(norm_plan)
            gate = model.plan_gate(norm_plan)
            # Shape: (B, 1, D) - Broadcastable across sequence length
            emb_bias = (gate * plan_emb).unsqueeze(1) 
            
            # B. Calculate KV Prefix (Past)
            kv = model.plan_to_kv(norm_plan)
            bsz = curr_plan.shape[0]
            kv = kv.view(
                bsz, 
                model.gpt2_n_layer, 
                2, 
                model.gpt2_n_head, 
                model.prefix_len, 
                model.gpt2_head_dim
            )
            kv = kv.permute(1, 2, 0, 3, 4, 5).contiguous() 
        
            # Initialize Cache with Plan
            past = DynamicCache()
            for l in range(model.gpt2_n_layer):
                past.update(kv[l, 0], kv[l, 1], l)
        
            # 2. Prepare Inputs (Manual Embedding Injection)
            # Get raw token embeddings
            tok_emb = model.gpt2_model.wte(hist_ids) 
            # Add the plan bias to every token in the history
            tok_emb = tok_emb + emb_bias 
        
            # 3. Forward Pass
            # Use inputs_embeds + past_key_values
            gpt2_output = model.gpt2_model.transformer(
                inputs_embeds=tok_emb,
                past_key_values=past,
                use_cache=True
            )
        
            last_hidden_state = gpt2_outputout.last_hidden_state[:, -1, :]
            next_token_logits = model.final_linear(last_hidden_state)
            return next_token_logits

        # ----- Per-step eval -----
        for t in range(1, T):
            valid = msk[:, t]  # token t must be real
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue

            hist_ids = ids[:, :t]                   # (N,t)
            tgt_ids  = ids[:, t]                    # (N,)


            logits_true = logits_from_prefix(plans_true_f, hist_ids)
            logits_shuf = logits_from_prefix(plans_shuf_f, hist_ids)
            logits_r1   = logits_from_prefix(plans_r1_f, hist_ids)
            logits_r2   = logits_from_prefix(plans_r2_f, hist_ids)

            # ---- NLL / ΔNLL (per-example then sum) ----
            nll_true_vec = F.cross_entropy(logits_true[valid], tgt_ids[valid], reduction="none")
            nll_shuf_vec = F.cross_entropy(logits_shuf[valid], tgt_ids[valid], reduction="none")
            delta_vec = nll_shuf_vec - nll_true_vec

            sum_nll_true[t] += nll_true_vec.sum()
            sum_nll_shuf[t] += nll_shuf_vec.sum()
            sum_delta[t] += delta_vec.sum()

            # ---- Accuracy ----
            pred_true = logits_true.argmax(dim=-1)
            pred_shuf = logits_shuf.argmax(dim=-1)
            acc_true_vec = (pred_true[valid] == tgt_ids[valid]).float()
            acc_shuf_vec = (pred_shuf[valid] == tgt_ids[valid]).float()

            sum_acc_true[t] += acc_true_vec.sum()
            sum_acc_shuf[t] += acc_shuf_vec.sum()

            # ---- KLs ----
            logp_true = F.log_softmax(logits_true[valid], dim=-1)
            p_true = logp_true.exp()
            logp_shuf = F.log_softmax(logits_shuf[valid], dim=-1)
            kl_ts_vec = _per_example_kl(p_true, logp_true, logp_shuf)
            sum_kl_ts[t] += kl_ts_vec.sum()

            logp_r1 = F.log_softmax(logits_r1[valid], dim=-1)
            p_r1 = logp_r1.exp()
            logp_r2 = F.log_softmax(logits_r2[valid], dim=-1)
            kl_rr_vec = _per_example_kl(p_r1, logp_r1, logp_r2)
            sum_kl_rr[t] += kl_rr_vec.sum()

            cnt[t] += n_valid

    eps = 1e-9
    counts = cnt.detach().cpu().long().tolist()

    mean_delta = (sum_delta / (cnt + eps)).detach().cpu().tolist()
    mean_nll_true = (sum_nll_true / (cnt + eps)).detach().cpu().tolist()
    mean_nll_shuf = (sum_nll_shuf / (cnt + eps)).detach().cpu().tolist()
    mean_kl_ts = (sum_kl_ts / (cnt + eps)).detach().cpu().tolist()
    mean_kl_rr = (sum_kl_rr / (cnt + eps)).detach().cpu().tolist()

    # Accuracy is already “sum correct / count”
    mean_acc_true = (sum_acc_true / (cnt + eps)).detach().cpu().tolist()
    mean_acc_shuf = (sum_acc_shuf / (cnt + eps)).detach().cpu().tolist()

    cutoff = 0
    for t in range(1, max_steps):
        if counts[t] >= min_count:
            cutoff = t

    return {
        "mean_delta_nll": mean_delta,
        "mean_nll_true": mean_nll_true,
        "mean_nll_shuf": mean_nll_shuf,
        "mean_acc_true": mean_acc_true,
        "mean_acc_shuf": mean_acc_shuf,
        "mean_kl_true_shuf": mean_kl_ts,
        "mean_kl_rand_rand": mean_kl_rr,
        "counts": counts,
        "cutoff_t": cutoff,
    }
