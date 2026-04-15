import torch
from contextlib import contextmanager

@contextmanager
def _eval_mode(model):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)

@torch.no_grad()
def encode_au_batch_in_chunks(
    au_batch: dict,
    inference_net,
    device,
    chunk_size: int = 8,
    use_amp: bool = False,
    ids_key: str = "enc_input_ids",
    mask_key: str = "enc_word_mask",
):
    """
    Takes a *big* CPU batch (au_batch) and runs inference_net on GPU in chunks.
    Returns:
      mu_t_all_cpu: (N, D)
      mu_i_all_cpu: (N, S, D)
    """
    ids = au_batch[ids_key]
    mask = au_batch[mask_key]
    N = ids.shape[0]

    mu_t_list = []
    mu_i_list = []

    amp_enabled = use_amp and (device.type == "cuda")

    with _eval_mode(inference_net):
        for start in range(0, N, chunk_size):
            end = min(N, start + chunk_size)

            ids_chunk  = ids[start:end].to(device, non_blocking=True)
            mask_chunk = mask[start:end].to(device, non_blocking=True)

            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    mu_t_q, _, mu_i_q, _ = inference_net(ids_chunk, mask_chunk)
            else:
                mu_t_q, _, mu_i_q, _ = inference_net(ids_chunk, mask_chunk)

            mu_t_list.append(mu_t_q.detach().cpu())
            mu_i_list.append(mu_i_q.detach().cpu())

            # free chunk GPU tensors ASAP
            del ids_chunk, mask_chunk, mu_t_q, mu_i_q

    mu_t_all_cpu = torch.cat(mu_t_list, dim=0)
    mu_i_all_cpu = torch.cat(mu_i_list, dim=0)
    return mu_t_all_cpu, mu_i_all_cpu