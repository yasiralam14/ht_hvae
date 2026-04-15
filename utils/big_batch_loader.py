import torch

def make_big_batch(loader, num_minibatches=50, device=None):
    """
    Concatenate `num_minibatches` minibatches from `loader` into one big batch.

    - Works for batches that are dict[str, Tensor] (most common).
    - If a value is not a Tensor, it will be collected into a list.
    - If `device` is provided, moves tensors onto that device.

    Returns:
      big_batch: dict with same keys as minibatch
    """
    it = iter(loader)
    batch_list = []

    for _ in range(num_minibatches):
        b = next(it)  # StopIteration if loader is too short

        if device is not None:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

        batch_list.append(b)

    # concatenate
    big = {}
    keys = batch_list[0].keys()

    for k in keys:
        v0 = batch_list[0][k]
        if torch.is_tensor(v0):
            # concat along batch dimension
            big[k] = torch.cat([b[k] for b in batch_list], dim=0)
        else:
            big[k] = [b[k] for b in batch_list]

    return big
