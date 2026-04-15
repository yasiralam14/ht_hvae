def _logits_next_from_plan_only(model, curr_plan, bos_id):
    """
    curr_plan: (N, d_model)
    bos_id: int
    returns logits: (N, V) for next token after prefix (and BOS)
    """
    bsz = curr_plan.size(0)
    device = curr_plan.device

    # 1. Prepare Plan (Same as above)
    norm_plan = model.plan_ln(curr_plan)
    
    plan_emb = model.plan_to_emb(norm_plan)
    gate = model.plan_gate(norm_plan)
    emb_bias = (gate * plan_emb).unsqueeze(1)

    kv = model.plan_to_kv(norm_plan)
    kv = kv.view(
        bsz, model.gpt2_n_layer, 2, model.gpt2_n_head,
        model.prefix_len, model.gpt2_head_dim
    )
    kv = kv.permute(1, 2, 0, 3, 4, 5).contiguous()

    past = DynamicCache()
    for l in range(model.gpt2_n_layer):
        past.update(kv[l, 0], kv[l, 1], l)

    # 2. Prepare Inputs (BOS Token + Injection)
    input_ids = torch.full((bsz, 1), bos_id, dtype=torch.long, device=device)
    tok_emb = model.gpt2_model.wte(input_ids)
    tok_emb = tok_emb + emb_bias

    # 3. Forward Pass
    gpt2_output = model.gpt2_model.transformer(
        inputs_embeds=tok_emb,
        past_key_values=past,
        use_cache=True
    )

    last_hidden_state = gpt2_output.last_hidden_state[:, -1, :]
    return model.final_linear(last_hidden_state)
