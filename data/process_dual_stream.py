def process_dual_stream(text, enc_tokenizer, dec_tokenizer, max_sentences, max_words):
    """
    Parses text into TWO sets of tensors (encoder + decoder), shaped as:
      [max_sentences][max_words]

    Encoder (DistilBERT):
      [CLS] sent_tokens [SEP] + pad

    Decoder (GPT-2):
      For each sentence:
        [BOS] sent_tokens [<EOS>]                (sentence terminator)
      For the VERY LAST sentence:
        [BOS] sent_tokens [<EOS>] [<|endoftext|>] (document terminator)

    Assumes:
      - Raw text contains "<BOS>" markers before each sentence and "<EOS>" delimiters
      - Raw text ends with "<|endoftext|>" (may appear once at end)
      - dec_tokenizer has bos_token="<BOS>", pad_token="<PAD>", and "<EOS>" added
    """

    # ----------------------------
    # 0) Resolve special token IDs
    # ----------------------------
    bos_id = dec_tokenizer.bos_token_id              # should be ID of "<BOS>"
    pad_id_dec = dec_tokenizer.pad_token_id          # should be ID of "<PAD>"
    eot_id = dec_tokenizer.eos_token_id              # GPT-2 original "<|endoftext|>"

    eos_sent_id = dec_tokenizer.convert_tokens_to_ids("<EOS>")  # your sentence EOS
    if eos_sent_id is None:
        raise ValueError('Decoder tokenizer does not know "<EOS>" (did you add_special_tokens?).')

    if bos_id is None or pad_id_dec is None or eot_id is None:
        raise ValueError("Decoder tokenizer is missing bos/pad/eos token ids. Check tokenizer setup.")

    # ----------------------------
    # 1) Clean + split into sentences
    # ----------------------------
    # Remove any explicit document terminator strings from the raw text;
    # we will add the doc terminator as an ID ourselves at the end.
    text = text.replace("<|endoftext|>", "").strip()

    raw_sentences = text.split("<EOS>")
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
        # Raw data prefixes each sentence with "<BOS>" as text; strip it
        if s.startswith("<BOS>"):
            s = s[len("<BOS>"):].strip()
        # In case of accidental repetition, strip repeatedly
        while s.startswith("<BOS>"):
            s = s[len("<BOS>"):].strip()
        if s:
            sentences.append(s)

    # ----------------------------
    # 2) Tokenize each sentence into rows
    # ----------------------------
    enc_ids_rows, enc_mask_rows = [], []
    dec_ids_rows, dec_mask_rows = [], []

    num_sentences = len(sentences)

    for i, sent in enumerate(sentences):
        # --- A) ENCODER (DistilBERT) ---
        enc_tokens = enc_tokenizer.encode(sent, add_special_tokens=False)
        enc_tokens = enc_tokens[: max_words - 2]  # reserve [CLS],[SEP]
        full_enc = [enc_tokenizer.cls_token_id] + enc_tokens + [enc_tokenizer.sep_token_id]

        enc_mask = [1] * len(full_enc)
        pad_len_enc = max_words - len(full_enc)
        if pad_len_enc > 0:
            full_enc += [enc_tokenizer.pad_token_id] * pad_len_enc
            enc_mask += [0] * pad_len_enc

        enc_ids_rows.append(full_enc)
        enc_mask_rows.append(enc_mask)

        # --- B) DECODER (GPT-2) ---
        dec_tokens = dec_tokenizer.encode(sent, add_special_tokens=False)

        is_last_sentence = (i == num_sentences - 1)

        if is_last_sentence:
            # Reserve 3 spots: BOS, <EOS>, <|endoftext|>
            dec_tokens = dec_tokens[: max_words - 3]
            full_dec = [bos_id] + dec_tokens + [eos_sent_id] + [eot_id]
        else:
            # Reserve 2 spots: BOS, <EOS>
            dec_tokens = dec_tokens[: max_words - 2]
            full_dec = [bos_id] + dec_tokens + [eos_sent_id]

        dec_mask = [1] * len(full_dec)
        pad_len_dec = max_words - len(full_dec)
        if pad_len_dec > 0:
            full_dec += [pad_id_dec] * pad_len_dec
            dec_mask += [0] * pad_len_dec

        dec_ids_rows.append(full_dec)
        dec_mask_rows.append(dec_mask)

    # ----------------------------
    # 3) Vertical padding to max_sentences
    # ----------------------------
    while len(enc_ids_rows) < max_sentences:
        enc_ids_rows.append([enc_tokenizer.pad_token_id] * max_words)
        enc_mask_rows.append([0] * max_words)

        dec_ids_rows.append([pad_id_dec] * max_words)
        dec_mask_rows.append([0] * max_words)

    # Truncate vertically if too many sentences
    enc_ids_rows = enc_ids_rows[:max_sentences]
    enc_mask_rows = enc_mask_rows[:max_sentences]
    dec_ids_rows = dec_ids_rows[:max_sentences]
    dec_mask_rows = dec_mask_rows[:max_sentences]

    return enc_ids_rows, enc_mask_rows, dec_ids_rows, dec_mask_rows
