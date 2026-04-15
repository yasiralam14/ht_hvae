import os
import torch
import wandb

def load_checkpoint_from_wandb(
    artifact_path,
    inference_net,
    generative_net,
    optimizer,
    scheduler=None,
    filename="hvae_checkpoint.pth"
):
    """
    Downloads a specific artifact from WandB, loads the state dictionaries,
    and returns the epoch to resume from.
    """
    print(f"Resuming from WandB artifact: {artifact_path}")

    # 1. Download the artifact
    # explicit run init is usually required if not already active,
    # but wandb.use_artifact works if run is active.
    artifact = wandb.use_artifact(artifact_path, type='model')
    artifact_dir = artifact.download()
    filepath = os.path.join(artifact_dir, filename)

    # 2. Load the file
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

    # 3. Restore states
    inference_net.load_state_dict(checkpoint['inference_state_dict'])
    generative_net.load_state_dict(checkpoint['generative_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 4. Return the next epoch
    # If we saved at epoch 9 (finished), we want to start at 10.
    start_epoch = checkpoint['epoch'] + 1
    val_loss = checkpoint.get('val_loss', 'N/A')

    print(f"Checkpoint loaded. Resuming from Epoch {start_epoch} (Last Val Loss: {val_loss})")
    return start_epoch