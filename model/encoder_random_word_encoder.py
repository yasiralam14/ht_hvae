import torch
import torch.nn as nn
from .mlp import MLPNetwork

class HT_HVAE_InferenceNetwork(nn.Module):
    """
    The Hierarchical Transformer Encoder (Inference Network) q(z|x).
    Now uses a randomly initialized Transformer for word-level encoding.
    """
    def __init__(self, hyperparams):
        super().__init__()

        self.latent_dim = hyperparams['latent_dim']
        self.d_model = hyperparams['d_model']
        self.vocab_size = hyperparams['vocab_size'] 
        self.max_sentences = hyperparams['max_sentences']
        self.max_words = hyperparams['max_words']
        self.n_heads = hyperparams['encoder_heads']
        self.dropout = hyperparams['encoder_dropout']
        self.n_layers = hyperparams['encoder_layers']
        self.pad_idx = hyperparams['pad_index']

        # --- 2.1. Word-Level Transformer Encoder (Random Init) ---

    
        self.word_embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_idx)

        
        self.word_position_embedding = nn.Embedding(self.max_words, self.d_model)

        # 3. Transformer Encoder Stack
        word_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=4 * self.d_model,
            dropout=self.dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.word_encoder = nn.TransformerEncoder(word_encoder_layer, num_layers=self.n_layers)

        # --- 2.2. Sentence-Level Transformer Encoder ---

        self.sentence_position_embedding = nn.Embedding(self.max_sentences + 1, self.d_model)

        sentence_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=4 * self.d_model,
            dropout=self.dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer_sentence = nn.TransformerEncoder(sentence_encoder_layer, num_layers=self.n_layers)

        # --- 2.3. Latent Variable Networks (MLP) ---

        # MLP for Global Posterior q(zt | x)
        self.mlp_global = MLPNetwork(self.d_model, self.latent_dim)
        self.local_latent = self.latent_dim

        # MLP for Local Posterior q(zi | xi)
        self.mlp_local = MLPNetwork(self.d_model, self.local_latent)

        # Learnable Document Token (similar to CLS for the whole document)
        self.doc_token = nn.Parameter(torch.randn(1, 1, self.d_model))

    def forward(self, input_ids, word_mask):
        """
        Args:
            input_ids: (batch_size, MAX_SENTENCES, MAX_WORDS) -> Pass dec_input_ids here
            word_mask: (batch_size, MAX_SENTENCES, MAX_WORDS) -> Pass dec_word_mask here
        """
        batch_size, max_sentences, max_words = input_ids.shape

        # --- 2.1. Word-Level Encoding (Shared Random Init Transformer) ---

        # Flatten: (B * N, W)
        flat_input_ids = input_ids.view(-1, max_words)

        # Create Boolean Mask for Transformer (True where padding exists)
        # word_mask is 1 for real, 0 for pad. Transformer needs True for Pad.
        flat_key_padding_mask = (word_mask.view(-1, max_words) == 0)

        # 1. Get Embeddings
        token_embeds = self.word_embedding(flat_input_ids) # (B*N, W, D)

        # 2. Add Positional Embeddings
        positions = torch.arange(0, max_words, device=input_ids.device).unsqueeze(0)
        pos_embeds = self.word_position_embedding(positions) # (1, W, D)

        hidden_states = token_embeds + pos_embeds

    
        word_encoder_output = self.word_encoder(
            src=hidden_states,
            src_key_padding_mask=flat_key_padding_mask
        )

        # 4. Extract Sentence Representation (Hw_0)
        
        H_w_0_flat = word_encoder_output[:, 0, :] # (B*N, D)

        # Handle purely padding sentences (firewall)
        is_padding_sentence = flat_key_padding_mask[:, 0]
        H_w_0_flat = H_w_0_flat.masked_fill(is_padding_sentence.unsqueeze(1), 0.0)

        # Reshape back: (batch_size, MAX_SENTENCES, D)
        H_w_0 = H_w_0_flat.view(batch_size, max_sentences, self.d_model)

        # --- 2.3. Local Posterior (q(zi | xi)) ---
        mu_i_flat, sigma2_i_flat = self.mlp_local(H_w_0_flat)
        mu_i = mu_i_flat.view(batch_size, max_sentences, self.local_latent)
        sigma2_i = sigma2_i_flat.view(batch_size, max_sentences, self.local_latent)

        # --- 2.2. Sentence-Level Encoding ---

        # Prepare inputs: Add Doc Token + Sentence Position Embeddings
        batch_doc_token = self.doc_token.expand(batch_size, -1, -1)
        H_sen_input_ = torch.cat([batch_doc_token, H_w_0], dim=1) # (B, N+1, D)

        position_ids = torch.arange(0, max_sentences + 1, device=input_ids.device)
        position_embeddings = self.sentence_position_embedding(position_ids)
        position_embeddings = position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        H_sen_input = H_sen_input_ + position_embeddings

        # Create Sentence Mask (True where sentence is padding)
        # sentence padding is where the first word is padded
        sentence_pad_mask = (word_mask[:, :, 0] == 0) # (B, N)
        doc_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=input_ids.device)
        full_sentence_mask = torch.cat([doc_mask, sentence_pad_mask], dim=1)

        # Sentence Transformer
        H_s = self.transformer_sentence(
            src=H_sen_input,
            src_key_padding_mask=full_sentence_mask
        )

        H_s_0 = H_s[:, 0, :] # (B, D)

        # --- 2.3. Global Posterior (q(zt | x)) ---
        mu_t, sigma2_t = self.mlp_global(H_s_0)

        return mu_t, sigma2_t, mu_i, sigma2_i