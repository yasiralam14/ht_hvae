# HT-HVAE — Hierarchical Transformer VAE for Long-Form Text Generation

> A PyTorch implementation of a two-level Hierarchical Variational Autoencoder (HVAE) for structured, multi-sentence text generation. The encoder is powered by **DistilBERT** for contextualized sentence representations; the decoder uses a **GRU sentence planner** combined with a **GPT-2 word generator** conditioned via KV-cache prefix injection.

---

## Overview

Generating coherent long-form text (e.g., structured abstracts) is hard: standard language models lack an explicit latent space that captures overall document structure. **HT-HVAE** addresses this by learning two levels of latent variables:

| Latent | Symbol | Captures |
|---|---|---|
| Global | `z_t` | Whole-document topic / style |
| Local | `z_i` | Per-sentence content / plan |

At each inference step the encoder produces a `(z_t, {z_i})` pair. At decoding, a GRU uses `z_t` and the sampled local latents to produce *plan vectors*, one per sentence. Each plan vector is then injected into GPT-2 via **learned KV-prefix tokens** — giving the language model soft, differentiable conditioning without any hard token concatenation.

---

## Architecture

```
Input Document  (B, N_sentences, N_words)
       │
       ▼
┌─────────────────────────────────────────────────┐
│        Inference Network  q(z | x)              │
│                                                 │
│  DistilBERT (frozen)  ──►  [CLS] per sentence  │
│         │                                       │
│  word_adapter (Linear, 768 → d_model)           │
│         │                                       │
│  TransformerEncoder (sentence-level, 2 layers)  │
│         │                                       │
│  ┌──────┴──────┐                               │
│  MLP_global    MLP_local                        │
│  μ_t, σ²_t    μ_i, σ²_i  (per sentence)        │
└─────────────────────────────────────────────────┘
       │                │
       ▼                ▼
    z_t ~ q           z_i ~ q     (reparameterization)
       │                │
       └────────┬───────┘
                ▼
┌─────────────────────────────────────────────────┐
│        Generative Network  p(x | z)             │
│                                                 │
│  GRU (init from z_t)  ◄── z_i sequences         │
│         │                                       │
│  plan_vectors (B, N, d_model)                   │
│         │                                       │
│  ┌──────┴──────────────────────────┐            │
│  plan_to_kv  ──► DynamicCache      │  ← prefix  │
│  plan_gate   ──► embedding bias    │  ← gate    │
│  └──────────────────────────────────┘           │
│                                                 │
│  GPT-2  (past_key_values = prefix_cache)        │
│         │                                       │
│  final_linear: [GPT-hidden ‖ plan_proj] → vocab │
└─────────────────────────────────────────────────┘
       │
       ▼
  Token Logits  (B, N, W, vocab_size)
```

### Key Design Choices

- **KV-cache prefix injection** — plan vectors are projected into all GPT-2 layers' key/value caches (prefix length = 5 virtual tokens). This gives the language model access to structural intent at every attention layer without polluting the actual token stream.  
- **Gated embedding bias** — a sigmoid-gated linear projection of the plan vector is added to every token embedding, providing a lightweight additive signal.  
- **Free-bits KL** — both global and local KL terms are clamped at 0.5 nats/dim, preventing posterior collapse early in training.  
- **Autoregressive local prior** — the prior `p(z_i | z_{i-1}, z_t)` is modelled by a small MLP conditioned on the previous local latent and the global latent, making the prior itself structured.  
- **Differential learning rates** — DistilBERT and GPT-2 backbones use `lr=2e-5`; all scratch components (GRU, MLPs, projections) use `lr=1e-4`.  
- **Word dropout** — at training time tokens are randomly replaced with `<MASK>` to regularize the decoder.

---

## Repository Structure

```
HVAE/
├── model/
│   ├── __init__.py
│   ├── encoder_distill_bert.py      # HT_HVAE_InferenceNetwork  (DistilBERT + Sentence Transformer)
│   ├── decoder_kv_cache.py          # HT_HVAE_GenerativeNetwork  (GRU + GPT-2 w/ KV prefix)
│   ├── mlp.py                       # Posterior MLP  (μ, σ²)
│   └── mlp_prior.py                 # Autoregressive local prior MLP
│
├── loss/
│   ├── __init__.py
│   └── loss.py                      # HT_HVAE_Loss  (recon + global KL + local KL w/ free-bits)
│
├── train/
│   ├── train.py                     # Entry point: data, optimiser, scheduler, training loop
│   └── train_one_epoch.py           # Single epoch pass with grad accumulation
│
├── data/
│   ├── __init__.py
│   ├── dataset_class.py             # PyTorch Dataset for structured abstracts
│   └── process_dual_stream.py       # Dual-tokenizer pre-processing (DistilBERT + GPT-2)
│
├── utils/
│   ├── __init__.py
│   ├── config.py                    # HyperParams dataclass  ← start here
│   ├── save_checkpoint.py
│   ├── load_saved_checkpoints.py
│   ├── reparamaterize.py
│   ├── get_kl_beta.py
│   ├── compute_active_units.py
│   └── ...
│
├── evaluations/                     # Inference scripts (NLL, sampling, oracle, SBERT, ...)
│   ├── plan_only_rollout_test.py
│   ├── hardest_of_k.py
│   ├── sbert_hardest_of_k.py
│   └── ...
│
├── hvae_kv_injection.ipynb          # Interactive notebook
└── pyproject.toml
```

---

## Hyperparameters

All hyperparameters live in `utils/config.py` as a dataclass:

| Parameter | Default | Description |
|---|---|---|
| `gpt2_model_name` | `"gpt2"` | GPT-2 variant for the decoder |
| `d_model` | `768` | Internal hidden dimension |
| `latent_dim` | `256` | Dimensionality of `z_t` and each `z_i` |
| `max_sentences` | `10` | Maximum sentences per document |
| `max_words` | `50` | Maximum tokens per sentence |
| `encoder_layers` | `2` | Sentence-level Transformer layers |
| `encoder_heads` | `8` | Attention heads in sentence encoder |
| `encoder_dropout` | `0.1` | Dropout in sentence encoder |
| `gru_layers` | `1` | GRU layers in sentence planner |
| `word_dropout_rate` | `0.0` | Probability of masking decoder input tokens |

---

## Dataset

The model is trained on the **arXiv Structured Abstracts** dataset (`data/arxiv_structured.parquet`, ~700 MB).
Each example contains a multi-section abstract broken into individual sentences, giving natural signal for the hierarchical structure.

> **Note:** The parquet file is excluded from this repository (too large). Download it separately and place it at `data/arxiv_structured.parquet`.

---

## Installation

```bash
git clone https://github.com/yasiralam14/ht_hvae.git
cd ht_hvae/HVAE
pip install -e .
```

`pyproject.toml` declares all dependencies:
`torch`, `transformers`, `sentence-transformers`, `scikit-learn`, `pandas`, `numpy`, `tqdm`, `wandb`, `nltk`, `beautifulsoup4`, `matplotlib`, `seaborn`.

---

## Training

```bash
cd HVAE/train
python train.py
```

Training uses:
- **Warmup + Cosine Annealing** scheduler (10 % warmup, cosine decay)
- **Gradient accumulation** to reach an effective batch size of 64
- **WandB** logging (set your key in `train.py`)
- **Checkpoint saving** every epoch via `utils/save_checkpoint.py`

To resume from a WandB artifact:
```python
WANDB_ARTIFACT_PATH = "your-entity/your-project/hvae-model:latest"
RESUME_TRAINING = True
```

---

## Loss Function

$$\mathcal{L} = \underbrace{\mathbb{E}_{q}\left[-\log p(x \mid z_t, \{z_i\})\right]}_{\text{reconstruction}} + \beta_g \underbrace{D_{\mathrm{KL}}\!\left[q(z_t \mid x) \,\|\, \mathcal{N}(0,I)\right]}_{\text{global KL}} + \beta_l \underbrace{\sum_i D_{\mathrm{KL}}\!\left[q(z_i \mid x) \,\|\, p(z_i \mid z_{i-1}, z_t)\right]}_{\text{local KL}}$$

Both KL terms use **free bits** (clamped at 0.5 nats/dim) to prevent posterior collapse.

---

## Inference / Evaluation

Several evaluation scripts are provided in `evaluations/`:

| Script | What it measures |
|---|---|
| `plan_only_rollout_test.py` | Generate full documents from latent samples |
| `hardest_of_k.py` | Oracle NLL: best of K samples vs ground truth |
| `sbert_hardest_of_k.py` | Oracle semantic similarity (SBERT) |
| `first_k_tokens_fed.py` | Partial-observation decoding (primed with K tokens)|
| `discriminate_unrestricted.py` | Latent-space discriminability |
| `latent_neighbours.py` | Nearest-neighbour navigation in latent space |
| `compute_active_units.py` | Active unit analysis (VAE posterior usage) |

---

## Dependencies

| Package | Role |
|---|---|
| `torch` | Core deep learning |
| `transformers` | DistilBERT encoder + GPT-2 decoder |
| `sentence-transformers` | SBERT for semantic evaluation |
| `wandb` | Experiment tracking |
| `pandas` | Data loading |
| `scikit-learn` | Train/validation split |
| `tqdm` | Progress bars |

---

## References

- Kingma & Welling. *Auto-Encoding Variational Bayes.* ICLR 2014.  
- Shen et al. *A Hierarchical Neural Autoencoder for Paragraphs and Documents.* ACL 2017.  
- Li et al. *Optimus: Organizing Sentences via Pre-trained Language Models.* EMNLP 2020.  
- Li et al. *Prefix-Tuning: Optimizing Continuous Prompts for Generation.* ACL 2021.  
- Lucas et al. *Don't Blame the ELBO! A Linear VAE Perspective on Posterior Collapse.* NeurIPS 2019.  

---

## License

MIT License. See `LICENSE` for details.
