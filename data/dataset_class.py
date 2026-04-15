import torch
from torch.utils.data import Dataset, DataLoader
from .process_dual_stream import process_dual_stream


class HVAEDataset(Dataset):
    def __init__(self, dataframe, enc_tokenizer, dec_tokenizer, max_sentences, max_words):
        self.data = dataframe['structured_abstract'].tolist()
        self.enc_tokenizer = enc_tokenizer # DistilBERT
        self.dec_tokenizer = dec_tokenizer # GPT-2
        self.max_sentences = max_sentences
        self.max_words = max_words

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]

        enc_ids, enc_mask, dec_ids, dec_mask = process_dual_stream(
            text,
            self.enc_tokenizer,
            self.dec_tokenizer,
            self.max_sentences,
            self.max_words,
        )

        return {
            'enc_input_ids': torch.tensor(enc_ids, dtype=torch.long),
            'enc_word_mask': torch.tensor(enc_mask, dtype=torch.long), # DistilBERT mask
            'dec_input_ids': torch.tensor(dec_ids, dtype=torch.long),
            'dec_word_mask': torch.tensor(dec_mask, dtype=torch.long)  # GPT-2 mask
        }

def create_dataloaders(df, enc_tokenizer,dec_tokenizer, hyperparams, batch_size=32):
    # 1. Create Dataset
    dataset = HVAEDataset(
        dataframe=df,
        enc_tokenizer=enc_tokenizer,
        dec_tokenizer = dec_tokenizer,
        max_sentences=hyperparams['max_sentences'],
        max_words=hyperparams['max_words'],
    )

    # 2. Create DataLoader
    # num_workers=0 is safer for debugging; increase for speed later
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    return dataloader