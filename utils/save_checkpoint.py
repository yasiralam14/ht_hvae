import os
import torch
import wandb

def save_checkpoint(
    inference_net,
    generative_net,
    optimizer,
    epoch,
    val_loss,
    scheduler=None,
    is_best=False,
    filename="hvae_checkpoint.pth",
):
    """
    Saves model state to a local file and uploads it to W&B as an artifact.
    - Writes: {filename}_epoch_{epoch}
    - Uploads that exact file (so W&B doesn't error).
    - Aliases: always "latest", plus "best" if is_best=True
    """
    # 1) Build state dict
    state = {
        "epoch": epoch,
        "inference_state_dict": inference_net.state_dict(),
        "generative_state_dict": generative_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "val_loss": float(val_loss) if hasattr(val_loss, "__float__") else val_loss,
    }

    # 2) Save locally
    save_path = f"{filename}_epoch_{epoch}"
    torch.save(state, save_path)

    # Sanity check (gives a clearer error than wandb's)
    if not os.path.isfile(save_path):
        raise FileNotFoundError(f"Checkpoint was not written: {save_path!r}")

    # 3) Create artifact (use a stable name so versions accumulate)
    artifact = wandb.Artifact(
        name="hvae-model",          # stable artifact name
        type="model",
        metadata={"epoch": epoch, "val_loss": state["val_loss"]},
    )

    # 4) Add the saved file
    # Store it in the artifact with a clean, consistent logical name
    artifact.add_file(save_path, name=filename)

    # 5) Log with aliases
    aliases = ["latest"]
    if is_best:
        aliases.append("best")

    wandb.log_artifact(artifact, aliases=aliases)

    # (Optional) return the path for convenience
    return save_path