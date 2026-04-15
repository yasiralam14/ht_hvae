import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from transformers.cache_utils import DynamicCache
from .mlp_prior import MLPNetworkForPrior

class HT_HVAE_GenerativeNetwork(nn.Module):
    """
    The Hierarchical VAE Generative Network (Decoder) p(x|z).
    Uses GRU for sentence planning and modified GPT-2 for word generation.
    """
    def __init__(self, hyperparams):
        super().__init__()

        self.latent_dim = hyperparams['latent_dim']
        self.local_latent = self.latent_dim
        self.d_model = hyperparams['d_model']
        self.gpt2_model_name = hyperparams['gpt2_model_name']
        self.vocab_size = hyperparams['vocab_size']
        self.gru_layers = hyperparams['gru_layers'] 

        # 3.2. Local Prior Network
        self.prior_mlp = MLPNetworkForPrior(self.latent_dim)
        self.gru_initial_projection = nn.Linear(self.latent_dim, self.d_model)

        # 3.3. Sentence-Level Decoder (GRU)
        self.gru = nn.GRU(
          input_size=self.local_latent,
          hidden_size=self.d_model,
          num_layers=self.gru_layers,
          batch_first=True
        )

        # 3.4. Load GPT-2
        config = GPT2Config.from_pretrained(self.gpt2_model_name)
        self.gpt2_model = GPT2Model.from_pretrained(self.gpt2_model_name, config=config)
        self.gpt2_model.resize_token_embeddings(self.vocab_size)
        self.gpt2_hidden_size = self.gpt2_model.config.n_embd

        # Resize embeddings to your vocab
        self.gpt2_model.resize_token_embeddings(self.vocab_size)
        gpt2_hidden_size = self.gpt2_model.config.n_embd

        cfg = self.gpt2_model.config
        self.gpt2_n_layer = cfg.n_layer
        self.gpt2_n_head  = cfg.n_head
        self.gpt2_n_embd  = cfg.n_embd
        assert self.gpt2_n_embd % self.gpt2_n_head == 0
        self.gpt2_head_dim = self.gpt2_n_embd // self.gpt2_n_head

        # choose prefix length for KV memory
        self.prefix_len = getattr(self, "prefix_len", 5)  # or pass as arg to __init__
        # (Optional but often helps) normalize plan before projections
        self.plan_ln = nn.LayerNorm(self.d_model)
        self.plan_to_emb = nn.Linear(self.d_model, self.gpt2_n_embd, bias=False)


        self.plan_gate = nn.Sequential(
                nn.Linear(self.d_model, 1),
                nn.Sigmoid()
        )

        hidden = 4 * self.gpt2_n_embd  # you can tune (2x, 4x, 8x)
        self.plan_to_kv = nn.Sequential(
            nn.Linear(self.d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.gpt2_n_layer * 2 * self.gpt2_n_head * self.prefix_len * self.gpt2_head_dim),
        )

        self.output_plan_proj = nn.Linear(self.d_model, self.gpt2_hidden_size)
        self.output_plan_ln = nn.LayerNorm(self.gpt2_hidden_size)

        # No need to concat plan here anymore; the prefix handles the "Mood"
        self.final_linear = nn.Linear(
          self.gpt2_hidden_size * 2, 
          self.vocab_size, 
          bias=False
        )

        # Hyperparameters
        self.word_dropout_rate = hyperparams['word_dropout_rate']
        self.mask_token_id = hyperparams.get('mask_token_id', self.gpt2_model.config.eos_token_id)




    def forward(self, input_ids, word_mask, z_t, z_i_samples, mu_i_prior=None, log_sigma2_i_prior=None):
        batch_size, max_sentences, max_words = input_ids.shape

        # --- 1. GRU Plan Generation ---
        # Initialize GRU with Global Latent z_t
        max_sentences = z_i_samples.size(1)
        z_t_expanded = z_t.unsqueeze(1).repeat(1, max_sentences, 1)

        # 2. Concatenate: GRU Input = [Local Latent; Global Latent]
        # Result shape: (Batch, Num_Sentences, Local_Dim + Global_Dim)
        gru_input = torch.cat([z_i_samples], dim=-1)

        # 3. Initialize Hidden State (Optional, but good to keep)
        h_0_projected = self.gru_initial_projection(z_t).unsqueeze(0)
        h_0 = h_0_projected.repeat(self.gru.num_layers, 1, 1)

        # 4. Pass Concatenated Input to GRU
        plan_vectors, _ = self.gru(gru_input, h_0)

        # --- 2. Prior Network (KL Logic unchanged) ---
        z_i_minus_1 = torch.cat([torch.zeros_like(z_i_samples[:, :1, :]), z_i_samples[:, :-1, :]], dim=1).view(-1, self.local_latent)
        z_t_flat = z_t.unsqueeze(1).repeat(1, max_sentences, 1).view(-1, self.latent_dim)
        mu_i_prior, sigma2_i_prior = self.prior_mlp(z_t_flat, z_i_minus_1)
        mu_i_prior = mu_i_prior.view(batch_size, max_sentences, self.local_latent)
        sigma2_i_prior = sigma2_i_prior.view(batch_size, max_sentences, self.local_latent)

        # --- 3. GENERATE PREFIXES (The New "Deep Injection") ---
        # Flatten plans: (B*N, D_MODEL)
        flat_plan_vectors = plan_vectors.reshape(-1, self.d_model)

        # Generate Virtual Tokens: (B*N, Prefix_Len, GPT_Hidden)
        # These are now normalized and scaled correctly by your class

        # --- 4. PREPARE TEXT INPUT ---
        flat_input_ids = input_ids.view(-1, max_words)
        bsz = flat_input_ids.size(0)


        # Word Dropout Logic
        if self.training and self.word_dropout_rate > 0:
            rand_mask = torch.rand(flat_input_ids.shape, device=input_ids.device)
            flat_word_mask_bool = word_mask.view(-1, max_words).bool()
            drop_mask = (rand_mask < self.word_dropout_rate) & flat_word_mask_bool
            input_ids_for_decoder = flat_input_ids.clone()
            input_ids_for_decoder[drop_mask] = self.mask_token_id
        else:
            input_ids_for_decoder = flat_input_ids

        

        tok_emb = self.gpt2_model.wte(input_ids_for_decoder)
        flat_plan_vectors = self.plan_ln(flat_plan_vectors)
        plan_emb = self.plan_to_emb(flat_plan_vectors)
        gate = self.plan_gate(flat_plan_vectors)
        tok_emb = tok_emb + (gate * plan_emb).unsqueeze(1) 

        kv = self.plan_to_kv(flat_plan_vectors)
        kv = kv.view(
        bsz,
        self.gpt2_n_layer,
        2,
        self.gpt2_n_head,
        self.prefix_len,
        self.gpt2_head_dim
    )

        kv = kv.permute(1, 2, 0, 3, 4, 5).contiguous()  # (L, 2, B, H, P, Hd)

        past = DynamicCache()
        for l in range(self.gpt2_n_layer):
            past.update(kv[l, 0], kv[l, 1], l)   # each: (B, H, P, Hd)

        flat_word_mask = word_mask.view(-1, max_words) # (B*N, Max_Words)
        prefix_mask = torch.ones(
            bsz, self.prefix_len,
            device=flat_word_mask.device,
            dtype=flat_word_mask.dtype
        )
        

        attention_mask = torch.cat([prefix_mask, flat_word_mask], dim=1)



        # --- 7. FORWARD PASS ---
        out = self.gpt2_model(
            inputs_embeds=tok_emb,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=False,
            return_dict=True,
    )

        # Get hidden states
        # Shape: (B*N, Prefix_Len + Max_Words, Hidden)
        text_hidden_states = out.last_hidden_state

        # --- 9. LOGITS ---
        # Compute logits only for text
        plan_feat = self.output_plan_ln(self.output_plan_proj(flat_plan_vectors))
        seq_len = text_hidden_states.size(1)
        plan_feat_expanded = plan_feat.unsqueeze(1).expand(-1, seq_len, -1)
        combined_features = torch.cat([text_hidden_states, plan_feat_expanded], dim=-1)
        reconstruction_logits = self.final_linear(combined_features)

        # Reshape back to batch structure
        reconstruction_logits = reconstruction_logits.view(
            batch_size, max_sentences, max_words, self.vocab_size
        )

        return reconstruction_logits, mu_i_prior, sigma2_i_prior

    def generate_autoregressive(
        self,
        z_t,
        z_i_samples,
        target_ids,
        eos_sentence_id,
        eos_doc_id,
        bos_token_id,
        pad_token_id,
        max_words=50,
        prime_len=0
    ):
        """
        Autoregressive generation with latent injection via 'past'.
        Assumes batch_size = 1.
        """
        self.eval()
        batch_size = 1
        batch, max_sentences, target_max_words = target_ids.shape

        # --- 1. REPLICATE PLAN GENERATION (Same as Forward) ---
        # Initialize GRU with Global Latent z_t
        z_t_expanded = z_t.unsqueeze(1).repeat(1, max_sentences, 1)

        # Concatenate: [Local; Global]
        gru_input = torch.cat([z_i_samples, z_t_expanded], dim=-1)

        # Initialize Hidden State
        h_0_projected = self.gru_initial_projection(z_t).unsqueeze(0)
        h_0 = h_0_projected.repeat(self.gru.num_layers, 1, 1)

        # Pass to GRU -> Get Plan Vectors
        # shape: (1, max_sentences, d_model)
        plan_vectors, _ = self.gru(gru_input, h_0)

        # --- 2. GENERATION LOOP ---
        all_generated_ids = [] # Will be list of lists (sentences -> words)
        all_logits = []
        all_predicted_ids = []
        stop_generation_flag = False

        with torch.no_grad():
            # Loop over sentences
            for sent_idx in range(max_sentences):
                if stop_generation_flag:
                    break

                # Get the specific latent plan for this sentence
                # Shape: (1, d_model) -> Needs to be compatible with GPT2 'past'
                # Based on your forward, we pass it directly.
                current_plan = plan_vectors[:, sent_idx, :]

                
                current_plan = self.plan_ln(current_plan)
                plan_emb = self.plan_to_emb(current_plan)
                gate = self.plan_gate(current_plan)
                embedding_bias = (gate * plan_emb).unsqueeze(1)

                kv = self.plan_to_kv(current_plan)
                kv = kv.view(
                    1,
                    self.gpt2_n_layer,
                    2,
                    self.gpt2_n_head,
                    self.prefix_len,
                    self.gpt2_head_dim
                )

                kv = kv.permute(1, 2, 0, 3, 4, 5).contiguous()  # (L, 2, B, H, P, Hd)

                past = DynamicCache()
                for l in range(self.gpt2_n_layer):
                    past.update(kv[l, 0], kv[l, 1], l)   # each: (B, H, P, Hd)


                # Start the sentence with BOS
                next_input_id = torch.tensor([[bos_token_id]], device=z_t.device)

                sent_generated_ids = [bos_token_id] # Track BOS in output
                sent_logits = []
                predicted_ids = []

                # Loop over words
                is_sent_finished = False
                for word_idx in range(max_words):

                    tok_emb = self.gpt2_model.wte(next_input_id)
                    tok_emb = tok_emb + embedding_bias
                    gpt2_output = self.gpt2_model(
                            inputs_embeds=tok_emb, 
                            past_key_values=past,
                            use_cache=True,
                            return_dict=True
                        )

                    past = gpt2_output.past_key_values

                    # Get last hidden state and project to vocab
                    last_hidden_state = gpt2_output.last_hidden_state[:, -1, :]
                    next_token_logits = self.final_linear(last_hidden_state) # (1, 1, Vocab)
                    sent_logits.append(next_token_logits.squeeze(0).squeeze(0)) # Store logits

                    # --- B. Decode & Priming Logic ---
                    predicted_id = torch.argmax(next_token_logits, dim=-1)
                    predicted_ids.append(predicted_id.item())
                    # Determine what the NEXT input will be
                    if word_idx < prime_len:
                        # K-Priming: Force the next input to be from targets
                        # Note: target_ids usually starts with BOS, so target_ids[..., 0] is BOS.
                        # If word_idx=0 (we just processed BOS), we want target_ids[..., 1].
                        # Ensure we don't go out of bounds of target_ids
                        if (word_idx + 1) < target_ids.size(2):
                            next_input_id = target_ids[:, sent_idx, word_idx + 1].view(1, 1)
                            actual_id_to_store = next_input_id.item()
                        else:
                            # Fallback if target is shorter than prime_len
                            next_input_id = predicted_id
                            actual_id_to_store = predicted_id.item()
                    else:
                        # Normal Greedy Decoding
                        next_input_id = predicted_id
                        actual_id_to_store = predicted_id.item()

                    sent_generated_ids.append(actual_id_to_store)
                    next_input_id = torch.tensor(
                            [[actual_id_to_store]], 
                            device=z_t.device
                        )

                    # --- C. Stopping Conditions ---

                    if actual_id_to_store == eos_doc_id:
                        stop_generation_flag = False
                        break # Stop words
                    
                    


                    if actual_id_to_store == eos_sentence_id:
                        is_sent_finished = True # Stop words (go to next sentence)

                # Append this sentence's results
                all_generated_ids.append(sent_generated_ids)
                all_logits.append(sent_logits)
                all_predicted_ids.append(predicted_ids)

        # --- 3. FORMAT OUTPUT ---
        # Because sentences might be different lengths, we usually pad them
        # to return a clean tensor, or return a list.
        # Here I will pad with a padding ID (assuming 0 or eos_doc) to match dimensions.

        # Find max length generated
        max_gen_len = max([len(s) for s in all_generated_ids])
        final_ids_tensor = torch.full((1, len(all_generated_ids), max_gen_len), pad_token_id, dtype=torch.long, device=z_t.device)

        max_pred_len = max([len(s) for s in all_predicted_ids])
        predicted_ids_tensor = torch.full((1, len(all_predicted_ids), max_pred_len), pad_token_id, dtype=torch.long, device=z_t.device)

        for i, s_ids in enumerate(all_predicted_ids):
            length = len(s_ids)
            predicted_ids_tensor[0, i, :length] = torch.tensor(s_ids, device=z_t.device)

        for i, s_ids in enumerate(all_generated_ids):
            length = len(s_ids)
            final_ids_tensor[0, i, :length] = torch.tensor(s_ids, device=z_t.device)

        vocab_size = self.vocab_size # Ensure this attribute exists
        final_logits_tensor = torch.zeros(
            (1, max_sentences, target_max_words, vocab_size),
            dtype=torch.float,
            device=z_t.device
        )

        for i, s_logits in enumerate(all_logits):
            if len(s_logits) > 0:
                # stack the list of tensors into (Seq_Len, Vocab)
                stacked_logits = torch.stack(s_logits)
                length = stacked_logits.size(0)
                # Fill the tensor, truncate if somehow longer than target (handled by loop limit, but safe to slice)
                limit = min(length, target_max_words)
                final_logits_tensor[0, i, :limit, :] = stacked_logits[:limit, :]

        return final_ids_tensor, all_generated_ids, final_logits_tensor, predicted_ids_tensor