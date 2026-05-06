import torch

def compute_transport_cost(T, sample_mu, n_samples=1024, device=device):
    x = sample_mu(n_samples).to(device)

    with torch.no_grad():
        Tx = T(x)

    # squared cost
    cost = ((Tx - x) ** 2).sum(dim=1).mean()

    return cost.item()

def sinkhorn_w2(x, y, epsilon=0.05, n_iter=100):
    n, m = x.shape[0], y.shape[0]

    # uniform weights
    a = torch.full((n,), 1.0 / n, device=x.device)
    b = torch.full((m,), 1.0 / m, device=y.device)

    # cost matrix (squared Euclidean)
    x_col = x.unsqueeze(1)   # (n,1,d)
    y_lin = y.unsqueeze(0)   # (1,m,d)
    C = torch.sum((x_col - y_lin) ** 2, dim=2)  # (n,m)

    # log-domain variables
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)

    for _ in range(n_iter):
        f = -epsilon * torch.logsumexp((g.unsqueeze(0) - C) / epsilon, dim=1) + epsilon * torch.log(a)
        g = -epsilon * torch.logsumexp((f.unsqueeze(1) - C) / epsilon, dim=0) + epsilon * torch.log(b)

    # transport plan (log)
    log_P = (f.unsqueeze(1) + g.unsqueeze(0) - C) / epsilon
    P = torch.exp(log_P)

    # Sinkhorn cost
    w2 = torch.sum(P * C)

    return w2

def compute_w2_sinkhorn(T, sample_mu, sample_nu,
                       n_samples=512, epsilon=0.05, device=device):
    x = sample_mu(n_samples).to(device)
    y_true = sample_nu(n_samples).to(device)

    with torch.no_grad():
        y_pred = T(x)

    w2 = sinkhorn_w2(y_pred, y_true, epsilon=epsilon)

    return w2.item()

def compute_w2_sinkhorn_noise(T, sample_mu, sample_nu, sigma = 0.05,
                       n_samples=512, epsilon=0.05, device=device):
    x = sample_mu(n_samples).to(device)
    z = torch.randn(n_samples, 2, device=device)
    x_tilde = x + sigma * z
    y_true = sample_nu(n_samples).to(device)

    with torch.no_grad():
        y_pred = T(x_tilde) 

    w2 = sinkhorn_w2(y_pred, y_true, epsilon=epsilon)

    return w2.item()