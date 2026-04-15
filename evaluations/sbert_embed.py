def sbert_embedder_wrapper(sbert_model, dec_tokenizer):
    """
    Creates an embed_fn that:
      1. Decodes the input IDs back to text strings (using dec_tokenizer).
      2. Re-encodes them using the SBERT model to get high-quality embeddings.
    """

    def embed_fn(ids_2d: torch.Tensor, mask_2d: torch.Tensor) -> torch.Tensor:
        # ids_2d: (N, W)

        # 1. Convert IDs -> Text (The Bridge)
        # We perform this on CPU because decode works on lists/strings
        ids_list = ids_2d.detach().cpu().tolist()

        # skip_special_tokens=True removes the padding and start/end tokens automatically
        sentences = dec_tokenizer.batch_decode(ids_list, skip_special_tokens=True)

        # 2. Encode with SBERT
        # convert_to_tensor=True gives us a torch tensor on the correct device
        embeddings = sbert_model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)

        return embeddings # Shape: (N, SBERT_Hidden_Dim)

    return embed_fn