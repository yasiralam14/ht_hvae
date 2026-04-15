import torch

def reparamaterize(mu, sigma2, eps=1e-8):
    sigma2 = torch.clamp(sigma2, min=eps)
    std = torch.sqrt(sigma2)
    return mu + torch.randn_like(std) * std