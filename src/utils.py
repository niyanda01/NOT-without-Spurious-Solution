import math
import matplotlib.pyplot as plt
import torch 

def cost(x, y):
    return ((x - y) ** 2).sum(dim=1, keepdim=True)/2

def show_mapping(T,sample_mu,sample_nu, device = "cpu"):
    x = sample_mu(1000).to(device)

    outputs = []
    for _ in range(5):
        outputs.append(T(x).detach().cpu())

    outputs = torch.cat(outputs, dim=0)
    x = x.cpu()
    y = sample_nu(1000).cpu()

    plt.figure()
    plt.scatter(x[:,0], x[:,1], label="mu", s=1, alpha=0.5)
    plt.scatter(y[:,0], y[:,1], label="nu", s=1, alpha=0.5)
    plt.scatter(outputs[:,0], outputs[:,1], label="T(x,z)", s=1, alpha=0.5)
    plt.axis('equal')
    plt.legend()
    plt.show()

def grad_norm(grads):
    return torch.sqrt(sum((g**2).sum() for g in grads))

def cosine_lr(step, total_steps, lr_max, lr_min=0.0):
    return lr_min + 0.5 * (lr_max - lr_min) * (
        1 + math.cos(math.pi * step / (step+total_steps))
    )

