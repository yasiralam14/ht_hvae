import torch.nn.utils as torch_utils
from tqdm import tqdm 
import wandb
from utils import (
    get_kl_beta,
    reparamaterize,
    encode_au_batch_in_chunks,
    compute_active_units
)



def train_one_epoch(
    inference_net,
    generative_net,
    loss_module,
    dataloader,
    optimizer,
    device,
    epoch_index,
    total_epochs,
    au_batch,
    scheduler = None
):
    inference_net.train()
    generative_net.train()

    total_epoch_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0

    local_b = 0.0
    if epoch_index >=4:
        local_b = 0.02


    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_index+1}")
    steps_per_epoch = len(dataloader)

    target_batch_size = 64
    batch_size = dataloader.batch_size # e.g., 8
    accumulation_steps = target_batch_size // batch_size

    optimizer.zero_grad()
    
    if epoch_index == 1:  # or whenever warm-up ends
        for p in generative_net.gpt2_model.parameters(): p.requires_grad = True
        for p in inference_net.word_encoder.parameters(): p.requires_grad = True





    for batch_idx, batch in enumerate(progress_bar):

        global_beta,local_beta = get_kl_beta(
                    epoch_index,
                    batch_idx,
                    steps_per_epoch,
                    MIN_BETA=0.005,  
                    MAX_BETA=0.5,    
                    CYCLE_EPOCHS=1, 
                    MIN_HOLD_FRAC=0.5, 
                    RAMP_FRAC=0.25      
      )
        # 1. Prepare Data
        # Assuming batch is a tuple/list; adjust unpacking based on your CollateFn
        enc_input_ids = batch['enc_input_ids'].to(device)
        enc_word_mask = batch['enc_word_mask'].to(device)
        dec_input_ids = batch['dec_input_ids'].to(device)
        dec_word_mask = batch['dec_word_mask'].to(device)

        # Target IDs are typically the same as input_ids for reconstruction
        # The Loss module should handle shifting (input[t] -> target[t+1]) internally
        # or via masking.
        target_ids = dec_input_ids.clone()


        # 2. Inference Network (Encoder)
        # Forward pass to get posterior parameters q(z|x)
        mu_t_q, sigma2_t_q, mu_i_q, sigma2_i_q = inference_net(enc_input_ids, enc_word_mask)

        # 3. Sampling (Reparameterization)
        z_t = reparamaterize(mu_t_q, sigma2_t_q)
        z_i_samples = reparamaterize(mu_i_q, sigma2_i_q)

        # 4. Generative Network (Decoder)
        # Reconstruct inputs based on samples and calculate priors p(z)
        reconstruction_logits, mu_i_p, sigma2_i_p = generative_net(
            dec_input_ids,
            dec_word_mask,
            z_t,
            z_i_samples
        )


        # 5. Loss Calculation
        loss, recon, global_kl, local_kl, kl_ratio, per_token_loss, global_klraw, local_raw, global_clamp_frac, local_clamp_frac = loss_module(
            # From Encoder (Posterior q)
            mu_t_q, sigma2_t_q, mu_i_q, sigma2_i_q,
            # From Decoder (Likelihood + Prior p)
            reconstruction_logits, mu_i_p, sigma2_i_p,
            # Targets
            target_ids, dec_word_mask,
            # Annealing
            local_kl_beta=global_beta,
            global_kl_beta = global_beta,

        )

        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        if batch_idx == 0 and epoch_index == 0:
          progress_bar.write(f"Global Mu Mean:   {mu_t_q.mean().item():.5f} | Std: {mu_t_q.std().item():.5f}")
          progress_bar.write(f"Global Sigma2 Mean: {sigma2_t_q.mean().item():.5f}")

        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient Clipping
            torch_utils.clip_grad_norm_(inference_net.parameters(), max_norm=1.0)
            torch_utils.clip_grad_norm_(generative_net.parameters(), max_norm=1.0)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad() # Reset for next set of accumulation



        current_lr_1 = optimizer.param_groups[0]['lr']
        current_lr_2 = optimizer.param_groups[1]['lr']

        # 7. Logging
        total_epoch_loss += loss.item()
        total_recon_loss += recon.item()
        # Sum global and local KL for display
        total_kl_loss += (global_kl.item() + local_kl.item())
        current_kl = global_kl.item() + local_kl.item()
        if batch_idx % 1000 == 0:


    
            log_dict = {
                "Losses/loss": loss.item(),
                "Losses/recon": recon.item(),
                "KL/kl_ratio": kl_ratio.item(),
                "KL/kl_local": local_kl.item(),
                "KL/kl_total": current_kl,
                "KL/kl_global": global_kl.item(),
                "KL/kl_global_raw": global_klraw.item(),
                "KL/kl_local_raw": local_raw.item(),
                "KL/kl_global_clamp_frac": global_clamp_frac.item(),
                "KL/kl_local_clamp_frac": local_clamp_frac.item(),
                "Losses/per_token_loss": per_token_loss.item(),
                'Betas/local_kl_beta':global_beta,
                'Betas/global_kl_beta': global_beta,
                'LRs/lr1': current_lr_1,
                'LRs/lr2': current_lr_2,
                }

    
    
            au_mu_t_q_cpu, au_mu_i_q_cpu = encode_au_batch_in_chunks(
                    au_batch,
                    inference_net,
                    device=device,
                    chunk_size=16,     # start small; increase until it fits
                    use_amp=False      # optional, often helps memory
                )
            au_metrics = compute_active_units(
                    au_mu_t_q_cpu,
                    au_mu_i_q_cpu,
                    threshold=0.01
                    )

            log_dict.update(au_metrics)
        
            wandb.log(log_dict)
