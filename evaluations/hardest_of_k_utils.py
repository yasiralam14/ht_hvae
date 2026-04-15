@torch.no_grad()
def _build_plans_from_latents(decoder, z_t, z_i_samples):
    """
    Reproduces your decoder's plan generation + prefix_generator part.
    Returns:
      prefix_embeds_flat: (B*S, prefix_len, gpt_hidden)
    """
    B, S, _ = z_i_samples.shape

    z_t_expanded = z_t.unsqueeze(1).expand(B, S, z_t.size(-1))
    gru_input = torch.cat([z_i_samples, z_t_expanded], dim=-1)  # (B,S, local+global)

    h0 = decoder.gru_initial_projection(z_t).unsqueeze(0)       # (1,B,H)
    h0 = h0.repeat(decoder.gru.num_layers, 1, 1)                # (L,B,H)

    plan_vectors, _ = decoder.gru(gru_input, h0)                # (B,S,d_model)
    flat_plan_vectors = plan_vectors.reshape(-1, decoder.d_model)  # (B*S, d_model)


    return flat_plan_vectors


@torch.no_grad()
def _nll_from_plan_pairs(
    decoder,
    compute_sentence_nll_avg,
    input_ids_flat,      # (N, W)
    word_mask_flat,      # (N, W)
    plan_flat,           # (N, d_model) - Assumed to be the latent vectors
    pair_chunk=256,
):
    """
    Runs the GPT2 forward with Deep Latent Injection (KV Prefix + Embedding Bias).
    Returns:
      nll: (N,)
      is_real: (N,)
    """
    device = input_ids_flat.device
    N, W = input_ids_flat.shape

    nll_out = torch.empty((N,), device=device, dtype=torch.float32)
    real_out = torch.empty((N,), device=device, dtype=torch.bool)

    for start in range(0, N, pair_chunk):
        end = min(start + pair_chunk, N)
        m = end - start

        ids = input_ids_flat[start:end]     # (m, W)
        msk = word_mask_flat[start:end]     # (m, W)
        curr_plan = plan_flat[start:end]    # (m, d_model)

        # 1. Prepare Plan (Norm & Projections)
        norm_plan = decoder.plan_ln(curr_plan)

        # A. Calculate Embedding Bias (Gate * Plan)
        # Shape: (m, 1, d_model) -> Broadcasts to all W tokens
        plan_emb = decoder.plan_to_emb(norm_plan)
        gate = decoder.plan_gate(norm_plan)
        emb_bias = (gate * plan_emb).unsqueeze(1) 

        # B. Calculate KV Prefix (Past)
        kv = decoder.plan_to_kv(norm_plan)
        kv = kv.view(
            m, 
            decoder.gpt2_n_layer, 
            2, 
            decoder.gpt2_n_head, 
            decoder.prefix_len, 
            decoder.gpt2_head_dim
        )
        kv = kv.permute(1, 2, 0, 3, 4, 5).contiguous() 

        # Initialize Cache with Plan
        past = DynamicCache()
        for l in range(decoder.gpt2_n_layer):
            past.update(kv[l, 0], kv[l, 1], l)

        # 2. Prepare Inputs (Manual Embedding Injection)
        # Get raw token embeddings
        tok_emb = decoder.gpt2_model.wte(ids) # (m, W, d_model)
        # Add bias to every token in the sequence
        tok_emb = tok_emb + emb_bias 

        # 3. Prepare Attention Mask
        # We must mask both the Prefix (already in 'past') and the Inputs
        prefix_mask = torch.ones(
            (m, decoder.prefix_len), 
            device=device, 
            dtype=msk.dtype
        )
        attn_mask = torch.cat([prefix_mask, msk], dim=1) # (m, P + W)

        # 4. Forward Pass
        # Returns hidden states for the input tokens only (W), attending to 'past'
        gpt2_output = decoder.gpt2_model.transformer(
            inputs_embeds=tok_emb,
            attention_mask=attn_mask,
            past_key_values=past,
            use_cache=True 
        )
        
        text_hidden = gpt2_output.last_hidden_state # (m, W, H)
        logits = decoder.final_linear(text_hidden) # (m, W, V)

        # 5. Compute NLL
        # compute_sentence_nll_avg expects (B, S, W, V) -> use S=1
        logits_4d = logits.unsqueeze(1)     # (m, 1, W, V)
        ids_3d   = ids.unsqueeze(1)         # (m, 1, W)
        msk_3d   = msk.unsqueeze(1)         # (m, 1, W)

        nll, is_real = compute_sentence_nll_avg(logits_4d, ids_3d, msk_3d)
        nll = nll.squeeze(1)                # (m,)
        is_real = is_real.squeeze(1)        # (m,)

        nll_out[start:end] = nll
        real_out[start:end] = is_real

    return nll_out, real_out
