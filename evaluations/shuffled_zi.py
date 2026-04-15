#Losses but with shuffled zt
enc_input_ids = first_batch['enc_input_ids'].to(device)
enc_word_mask = first_batch['enc_word_mask'].to(device)
dec_input_ids = first_batch['dec_input_ids'].to(device)
dec_word_mask = first_batch['dec_word_mask'].to(device)

# Target IDs are typically the same as input_ids for reconstruction
# The Loss module should handle shifting (input[t] -> target[t+1]) internally
# or via masking.
target_ids = dec_input_ids.clone()


# 2. Inference Network (Encoder)
# Forward pass to get posterior parameters q(z|x)
mu_t_q, sigma2_t_q, mu_i_q, sigma2_i_q = inference_net(dec_input_ids, dec_word_mask)

# 3. Sampling (Reparameterization)
z_t = reparameterize(mu_t_q, sigma2_t_q)
z_i_samples = reparameterize(mu_i_q, sigma2_i_q)

perm = torch.randperm(z_t.size(0))
z_t, z_i_samples_permed = z_t, z_i_samples[perm]

# 4. Generative Network (Decoder)
# Reconstruct inputs based on samples and calculate priors p(z)
reconstruction_logits, mu_i_p, sigma2_i_p = generative_net(
    dec_input_ids,
    dec_word_mask,
    z_t,
    z_i_samples_permed
)

# 5. Loss Calculation
loss, recon, global_kl, local_kl, kl_ratio, per_token_loss = loss_module(
    # From Encoder (Posterior q)
    mu_t_q, sigma2_t_q, mu_i_q, sigma2_i_q,
    # From Decoder (Likelihood + Prior p)
    reconstruction_logits, mu_i_p, sigma2_i_p,
    # Targets
    target_ids, dec_word_mask,
    # Annealing
    local_kl_beta=1,
    global_kl_beta = 1,

)