def calculate_nll_ignore_index(logits, real_ids, pad_token_id):
    # Pass ignore_index directly
    loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='mean')

    # Flatten inputs
    # logits: (Batch * Sent * Words, Vocab)
    # real_ids: (Batch * Sent * Words)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = real_ids[..., 1:].contiguous()
    nll = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    return nll