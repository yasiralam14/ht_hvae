def _compute_plans_flat(model, encoder, dec_input_ids, dec_word_mask, use_mu=True, shuffle_global=False):
    """
    Returns flattened per-sentence:
      ids, msk: (N,W)
      plan_true, plan_shuf: (N,D_model)
    where N = (# real sentences in batch)
    """
    device = dec_input_ids.device
    B, S, W = dec_input_ids.shape

    mu_t, logvar_t, mu_i, logvar_i = encoder(dec_input_ids, dec_word_mask)
    if use_mu:
        z_t, z_i = mu_t, mu_i
    else:
        z_t = mu_t + torch.randn_like(mu_t) * torch.exp(0.5 * logvar_t)
        z_i = mu_i + torch.randn_like(mu_i) * torch.exp(0.5 * logvar_i)

    z_t_exp = z_t.unsqueeze(1).repeat(1, S, 1)
    h0 = model.gru_initial_projection(z_t).unsqueeze(0).repeat(model.gru.num_layers, 1, 1)

    # true plans
    plans_true, _ = model.gru(torch.cat([z_i, z_t_exp], dim=-1), h0)  # (B,S,Dm)

    # shuffled plans
    perm = torch.randperm(B, device=device)
    z_i_shuf = z_i[perm]
    if shuffle_global:
        z_t_shuf = z_t[perm]
        z_t_exp_shuf = z_t_shuf.unsqueeze(1).repeat(1, S, 1)
    else:
        z_t_exp_shuf = z_t_exp
    plans_shuf, _ = model.gru(torch.cat([z_i_shuf, z_t_exp_shuf], dim=-1), h0)

    # flatten sentences
    ids = dec_input_ids.reshape(B * S, W)
    msk = dec_word_mask.reshape(B * S, W).bool()
    real_sent = msk.sum(dim=1) > 0
    if real_sent.sum().item() == 0:
        return None

    ids = ids[real_sent]
    msk = msk[real_sent]
    plan_true = plans_true.reshape(B * S, -1)[real_sent]
    plan_shuf = plans_shuf.reshape(B * S, -1)[real_sent]
    return ids, msk, plan_true, plan_shuf

