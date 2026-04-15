from dataclasses import dataclass, asdict

@dataclass()
class HyperParams:
    # --- Architecture Constraints ---
    gpt2_model_name: str = "gpt2"
    d_model: int = 768
    vocab_size: int = 50257

    # --- Latent Space ---
    latent_dim: int = 256

    # --- Data Dimensions ---
    max_sentences: int = 10
    max_words: int = 50

    # --- Inference Network (Encoder) ---
    encoder_layers: int = 2
    encoder_heads: int = 8
    encoder_dropout: float = 0.1

    # --- Sentence Decoder (GRU) ---
    gru_layers: int = 1

    # --- Special Tokens ---
    pad_index: int = 50256
    word_dropout_rate: float = 0.0
    plan_dropout_rate: float = 0.0
    mask_token_id: int = 10

HP = HyperParams()
HP_DICT = asdict(HP)
