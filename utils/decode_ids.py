

def decode_original_batch(input_ids, tokenizer):
    """
    Decodes a 3D tensor of input_ids into text.

    Args:
        input_ids (torch.Tensor): Shape (Batch_Size, Max_Sentences, Max_Words)
        tokenizer: HuggingFace tokenizer

    Returns:
        list[list[str]]: Nested list of strings [Batch][Sentence]
    """
    # Ensure it's on CPU and convert to list if it's a tensor
    if hasattr(input_ids, 'cpu'):
        input_ids = input_ids.cpu()

    batch_size, num_sentences, max_words = input_ids.shape
    all_decoded_text = []

    for b in range(batch_size):
        batch_sentences = []
        print(f"\n--- Batch Sample {b} ---")

        for n in range(num_sentences):
            # Get the sequence for this specific sentence
            sequence = input_ids[b, n]

            # Remove padding (optional, depending on tokenizer behavior)
            # sequence = sequence[sequence != tokenizer.pad_token_id]

            text = tokenizer.decode(sequence, skip_special_tokens=True)

            # Filter out empty strings if entire row was padding
            if text.strip():
                batch_sentences.append(text)

        all_decoded_text.append(batch_sentences)

    return all_decoded_text