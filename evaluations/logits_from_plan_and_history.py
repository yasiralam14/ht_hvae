def _logits_next_from_plan_and_history(model, curr_plan, history_ids):
    """
    curr_plan: (N, d_model) - Raw latent plan
    history_ids: (N, t) - Sequence of history token IDs
    returns logits: (N, V) for next token
    """
    bsz = curr_plan.size(0)

    # 1. Prepare Plan: Norm & Projections
    norm_plan = model.plan_ln(curr_plan)
    
    # A. Calculate Embedding Bias (Gate * Plan)
    plan_emb = model.plan_to_emb(norm_plan)
    gate = model.plan_gate(norm_plan)
    emb_bias = (gate * plan_emb).unsqueeze(1) # (N, 1, d_model)

    # B. Calculate KV Prefix (Past)
    kv = model.plan_to_kv(norm_plan)
    kv = kv.view(
        bsz, model.gpt2_n_layer, 2, model.gpt2_n_head,
        model.prefix_len, model.gpt2_head_dim
    )
    kv = kv.permute(1, 2, 0, 3, 4, 5).contiguous()

    # Initialize Cache
    past = DynamicCache()
    for l in range(model.gpt2_n_layer):
        past.update(kv[l, 0], kv[l, 1], l)

    # 2. Prepare Inputs (Manual Embedding Injection)
    tok_emb = model.gpt2_model.wte(history_ids) 
    tok_emb = tok_emb + emb_bias # Inject plan into history embeddings

    # 3. Forward Pass
    gpt2_output = model.gpt2_model.transformer(
        inputs_embeds=tok_emb,
        past_key_values=past,
        use_cache=True
    )
    
    last_hidden_state = gpt2_output.last_hidden_state[:, -1, :]
    return model.final_linear(last_hidden_state)

