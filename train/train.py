import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import GPT2Tokenizer, DistilBertTokenizer

# Import your models and loss
from model import HT_HVAE_InferenceNetwork, HT_HVAE_GenerativeNetwork
from loss import HT_HVAE_Loss

# Import your training and checkpointing functions
from utils import (
    save_checkpoint,
    load_checkpoint_from_wandb,
    HP_DICT,
    make_big_batch
)
from train_one_epoch import train_one_epoch

from data import create_dataloaders

wandb.login(key="0ce56922c7ea30310a87d49246b15bc7d7ca9c89")

hyperParams = HP_DICT


# Starting wandb project
wandb.init(
    project="hvae_frozen_min_beta",
    config=hyperParams,
    name = 'only_plan_10epochs'
)



# adding the special tokens to the tokenizer
# 1. Load standard tokenizers
dec_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
enc_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# 2. Add Special Tokens
# We use 'additional_special_tokens' for custom tokens like EOT
special_tokens_dict = {
    "bos_token": "<BOS>",          
    "pad_token": "<PAD>",         

    "additional_special_tokens": [
        "<EOS>",                   
        "<MASK>",                  
    ],
}

dec_tokenizer.add_special_tokens(special_tokens_dict)
# 3. Get IDs
# Standard attributes exist for bos/eos/pad/mask
pad_idx = dec_tokenizer.pad_token_id                 # <PAD>
bos_idx = dec_tokenizer.bos_token_id                 # <BOS>

# GPT-2's original eos_token (<|endoftext|>)
gpt2_eos_id = dec_tokenizer.eos_token_id

# Your custom tokens
eos_idx = dec_tokenizer.convert_tokens_to_ids("<EOS>")
mask_idx = dec_tokenizer.convert_tokens_to_ids("<MASK>")

new_vocab_size = len(dec_tokenizer)

print(f"New Vocab Size: {new_vocab_size}")
print(f"PAD ID: {pad_idx} | BOS ID: {bos_idx}")
print(f"GPT2 EOS ID (<|endoftext|>): {gpt2_eos_id}")
print(f"Custom EOS ID (<EOS>): {eos_idx}")
print(f"MASK ID: {mask_idx}")

# 3. Update hyperparameters
hyperParams['vocab_size'] = new_vocab_size
hyperParams['pad_index'] = pad_idx
hyperParams['bos_index'] = bos_idx
hyperParams['gpt2_eos_id'] = gpt2_eos_id   # optional, but nice to keep
hyperParams['eos_token_id'] = eos_idx      # your custom EOS
hyperParams['mask_token_id'] = mask_idx


df = pd.read_parquet("/home/salam4/ht_hvae/HVAE/data/arxiv_structured.parquet")
df["structured_abstract"] = df["structured_abstract"].str.replace(
    "<endoftext>",
    "<|endoftext|>",
    regex=False
)

train_df, valid_df = train_test_split(df, test_size=0.01, random_state=42)





# Creating dataloaders
train_subset = train_df.sample(frac=1, random_state=42)
test_subset = valid_df.sample(frac=0.01, random_state=42)
train_loader = create_dataloaders(train_subset, enc_tokenizer,dec_tokenizer, hyperParams, batch_size=32)
valid_loader = create_dataloaders(test_subset, enc_tokenizer,dec_tokenizer ,hyperParams, batch_size=8)


au_batch = make_big_batch(train_loader, num_minibatches=50)

# Models initialization + training is done here
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WANDB_ARTIFACT_PATH = "yasir-alam14/HVAE-kv_injection_full_data_distillbert_frozen/hvae-model:latest" # Set this to None if starting fresh
RESUME_TRAINING = False

num_epochs = 10
# 1. Calculate total training steps
target_batch_size = 64
batch_size = train_loader.batch_size 
accumulation_steps = target_batch_size // batch_size

total_steps = (num_epochs * len(train_loader)) // accumulation_steps

# 2. Define Warmup (e.g., 5% of total steps)
warmup_steps = int(0.1 * total_steps)
decay_steps = total_steps - warmup_steps

print(f"Total Steps: {total_steps} | Warmup Steps: {warmup_steps}")

# Instantiate Models
inference_net = HT_HVAE_InferenceNetwork(hyperParams).to(device)
generative_net = HT_HVAE_GenerativeNetwork(hyperParams).to(device)
loss_module = HT_HVAE_Loss(hyperParams).to(device)


# Define your Learning Rates
LR_GPT2 = 2e-5   # Low LR for pre-trained weights
LR_BERT = 2e-5
LR_REST = 1e-4   # High LR for new Encoders/GRU/MLPs

gpt2_params = list(generative_net.gpt2_model.parameters())
distilbert_params = list(inference_net.word_encoder.parameters())

# 2. Identify "Everything Else"
# Create a set of IDs for parameters already assigned to avoid duplication
assigned_ids = set(map(id, gpt2_params + distilbert_params))
scratch_params = []

# Iterate through both networks and collect unassigned parameters
for model in [inference_net, generative_net]:
    for param in model.parameters():
        if id(param) not in assigned_ids:
            scratch_params.append(param)

# 3. Create 3-Group Optimizer
optimizer = torch.optim.AdamW([
    # No weight decay for pre-trained models to preserve embedding norms
    {'params': gpt2_params, 'lr': LR_GPT2, 'weight_decay': 0.0},
    {'params': distilbert_params, 'lr': LR_BERT, 'weight_decay': 0.0},

    # Keep weight decay for scratch parameters to prevent them from exploding
    {'params': scratch_params, 'lr': LR_REST, 'weight_decay': 0.00}
])


scheduler_warmup = LinearLR(
    optimizer,
    start_factor=0.01,
    end_factor=1.0,
    total_iters=warmup_steps
)


scheduler_decay = CosineAnnealingLR(
    optimizer,
    T_max=decay_steps,
    eta_min=1e-6
)

scheduler = SequentialLR(
    optimizer,
    schedulers=[scheduler_warmup, scheduler_decay],
    milestones=[warmup_steps]
)

start_epoch = 0

if RESUME_TRAINING and WANDB_ARTIFACT_PATH:
    if wandb.run is None:
        wandb.init(project="my_project", resume="allow")

    start_epoch = load_checkpoint_from_wandb(
        artifact_path=WANDB_ARTIFACT_PATH,
        inference_net=inference_net,
        generative_net=generative_net,
        optimizer=optimizer,
        scheduler=scheduler
    )


special_tokens_set = {'<PAD>', '<BOS>', '<EOS>', '<|endoftext|>'}

for p in generative_net.gpt2_model.parameters(): p.requires_grad = False
for p in inference_net.word_encoder.parameters(): p.requires_grad = False


for epoch in range(start_epoch, num_epochs):
    if epoch >= num_epochs:
        print("Training already completed for this number of epochs.")
        break
    train_one_epoch(inference_net, generative_net, loss_module, train_loader, optimizer, device, epoch, num_epochs, au_batch,scheduler)
    #results = evaluate_model(inference_net, generative_net, loss_module, valid_loader, tokenizer, device, special_tokens_set)
    #validation_logs = {f"validation/{k}": v for k, v in results.items()}
    #wandb.log(validation_logs)
    if (epoch + 1 )%1 == 0:
        save_checkpoint(
            inference_net=inference_net,
            generative_net=generative_net,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=0.1,
            scheduler=scheduler,
            is_best=False
        )
    