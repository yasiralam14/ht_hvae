def greedy_decode_with_oracle_prefix(decoder, curr_plans, gold_ids, gold_mask,
                                    K, max_new_tokens=200,
                                    bos_id=50256, eos_id=50256):
    """
    Returns token-accuracy over the remaining gold tokens after K.
    """
    device = gold_ids.device
    
    # --- FIX 1: Define Batch Size 'm' ---
    m = curr_plans.size(0) 

    L = int(gold_mask.sum().item())
    if L <= 1:
        return None  # too short

    gold = gold_ids[:L]  # (L,)
    K = int(min(K, L-1))  # ensure at least 1 target token remains
    
    # Target to evaluate against
    target = gold[K:] 
    if target.numel() == 0:
        return 0.0 # No tokens to predict

    # Context: [BOS] + gold[:K]
    ctx_ids = torch.cat([torch.tensor([bos_id], device=device), gold[:K]], dim=0).unsqueeze(0)  # (1, 1+K)

    # 1. Prepare Plan & KV (Deep Injection)
    norm_plan = decoder.plan_ln(curr_plans)

    # A. Embedding Bias
    plan_emb = decoder.plan_to_emb(norm_plan)
    gate = decoder.plan_gate(norm_plan)
    emb_bias = (gate * plan_emb).unsqueeze(1) # (m, 1, H)

    # B. KV Prefix
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

    past = DynamicCache()
    for l in range(decoder.gpt2_n_layer):
        past.update(kv[l, 0], kv[l, 1], l)

    # 2. Initial Forward Pass (Process Oracle Context)
    tok_emb = decoder.gpt2_model.wte(ctx_ids) # (m, 1+K, H)
    tok_emb = tok_emb + emb_bias 

    gpt2_output = decoder.gpt2_model.transformer(
        inputs_embeds=tok_emb,
        past_key_values=past,
        use_cache=True,
        return_dict=True
    )
    
    # --- FIX 2: Capture initial 'past' ---
    past = gpt2_output.past_key_values 
    
    last_h = gpt2_output.last_hidden_state[:,-1,:] 
    
    preds = []
    # Predict first token
    logits = decoder.final_linear(last_h)
    next_tok = torch.argmax(logits, dim=-1)
    preds.append(next_tok.item())

    # 3. Generation Loop
    T = target.size(0)
    
    # We loop until we match target length or max tokens
    for _ in range(min(max_new_tokens, T-1)):
        if preds[-1] == eos_id:
            break
            
        # --- FIX 3: Correct Input Shape (1,1) ---
        ctx_ids = torch.tensor([[preds[-1]]], device=device, dtype=torch.long) # Shape (1, 1)

        tok_emb = decoder.gpt2_model.wte(ctx_ids) 
        tok_emb = tok_emb + emb_bias 

        gpt2_output = decoder.gpt2_model.transformer(
            inputs_embeds=tok_emb,
            past_key_values=past, # Pass history
            use_cache=True,
            return_dict=True
        )
        
        # --- FIX 4: Update 'past' ---
        past = gpt2_output.past_key_values 
        
        last_h = gpt2_output.last_hidden_state[:, -1, :]
        logits = decoder.final_linear(last_h)
        next_tok = torch.argmax(logits, dim=-1)
        preds.append(next_tok.item())

    # Score
    preds_t = torch.tensor(preds, device=device, dtype=torch.long)
    m_len = min(preds_t.numel(), target.numel())
    correct = (preds_t[:m_len] == target[:m_len]).sum().item()
    total = target.numel()
    
    return correct / total