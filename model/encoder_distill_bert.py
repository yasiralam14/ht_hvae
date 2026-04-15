import torch
import torch.nn as nn
from transformers import AutoModel
from .mlp import MLPNetwork

class HT_HVAE_InferenceNetwork(nn.Module):
    """
    The Hierarchical Transformer Encoder (Inference Network) q(z|x).
    Now uses pre-trained DistilRoBERTa for word-level encoding.
    """
    def __init__(self, hyperparams):
        super().__init__()

        self.latent_dim = hyperparams['latent_dim']
        self.d_model = hyperparams['d_model']
        self.max_sentences = hyperparams['max_sentences']
        self.n_heads = hyperparams['encoder_heads']
        self.dropout = hyperparams['encoder_dropout']
        self.n_layers = hyperparams['encoder_layers']

        # --- 2.1. Word-Level Transformer Encoder (DistilRoBERTa) ---
        
        # Load pre-trained model
        self.word_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        
        # Projection layer to map DistilRoBERTa's 768 dim to your d_model
        self.distilroberta_dim = 768
        self.word_adapter = nn.Linear(self.distilroberta_dim, self.d_model)

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
            input_ids: (batch_size, MAX_SENTENCES, MAX_WORDS) -> Tokenized with DistilRoBERTa tokenizer
            word_mask: (batch_size, MAX_SENTENCES, MAX_WORDS) -> 1 for real tokens, 0 for pad
        """
        batch_size, max_sentences, max_words = input_ids.shape

        # --- 2.1. Word-Level Encoding (DistilRoBERTa) ---

        # Flatten: (B * N, W)
        flat_input_ids = input_ids.view(-1, max_words)
        flat_attention_mask = word_mask.view(-1, max_words)

        # Pass through DistilRoBERTa
        # DistilRoBERTa handles its own positional embeddings
        outputs = self.word_encoder(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask
        )

        # Extract [CLS] token representation (Index 0 for RoBERTa family is <s>)
        # last_hidden_state shape: (B*N, W, 768)
        cls_vectors = outputs.last_hidden_state[:, 0, :] # (B*N, 768)

        # Project to d_model space to match the rest of the network
        H_w_0_flat = self.word_adapter(cls_vectors) # (B*N, d_model)

        # Handle purely padding sentences (firewall)
        # Identify sentences where the first token is masked (padding sentence)
        is_padding_sentence = (flat_attention_mask[:, 0] == 0)
        H_w_0_flat = H_w_0_flat.masked_fill(is_padding_sentence.unsqueeze(1), 0.0)

        # --- 2.3. Local Posterior (q(zi | xi)) ---
        mu_i_flat, sigma2_i_flat = self.mlp_local(H_w_0_flat)
        mu_i = mu_i_flat.view(batch_size, max_sentences, self.local_latent)
        sigma2_i = sigma2_i_flat.view(batch_size, max_sentences, self.local_latent)

        # --- 2.2. Sentence-Level Encoding ---
        
        # Reshape H_w_0 back for sentence encoder
        H_w_0 = H_w_0_flat.view(batch_size, max_sentences, self.d_model)

        # Prepare inputs: Add Doc Token + Sentence Position Embeddings
        batch_doc_token = self.doc_token.expand(batch_size, -1, -1)
        H_sen_input_ = torch.cat([batch_doc_token, H_w_0], dim=1) # (B, N+1, D)

        position_ids = torch.arange(0, max_sentences + 1, device=input_ids.device)
        position_embeddings = self.sentence_position_embedding(position_ids)
        position_embeddings = position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        H_sen_input = H_sen_input_ + position_embeddings

        # Create Sentence Mask (True where sentence is padding)
        # In PyTorch TransformerEncoder, src_key_padding_mask requires True for Padding
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