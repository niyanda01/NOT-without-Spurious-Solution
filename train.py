import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ======================
# Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# Data Sampling
# ======================
def sample_mu(batch_size):
    return torch.randn(batch_size, 2).to(device)


def sample_nu(batch_size):
    # example: circle distribution
    theta = torch.rand(batch_size) * 2 * np.pi
    x = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    return x.to(device)


# ======================
# Models
# ======================
class Transport(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


# ======================
# Cost
# ======================
def cost(x, y):
    return ((x - y) ** 2).sum(dim=1)


# ======================
# Training
# ======================
def train():
    T = Transport().to(device)
    f = Critic().to(device)

    optimizer_T = optim.Adam(T.parameters(), lr=1e-3)
    optimizer_f = optim.Adam(f.parameters(), lr=1e-3)

    steps = 20000
    batch_size = 256

    for step in range(steps):
        x = sample_mu(batch_size)
        y = sample_nu(batch_size)

        # ===== Critic update =====
        optimizer_f.zero_grad()
        loss_f = -(f(T(x)).mean() - f(y).mean())
        loss_f.backward()
        optimizer_f.step()

        # ===== Transport update =====
        optimizer_T.zero_grad()
        loss_T = cost(x, T(x)).mean() - f(T(x)).mean()
        loss_T.backward()
        optimizer_T.step()

        if step % 1000 == 0:
            print(f"Step {step} | Loss_T: {loss_T.item():.4f} | Loss_f: {loss_f.item():.4f}")

    return T


# ======================
# Visualization
# ======================
def visualize(T):
    x = sample_mu(1000).detach()
    Tx = T(x).detach()

    plt.scatter(x[:, 0].cpu(), x[:, 1].cpu(), alpha=0.3, label="mu")
    plt.scatter(Tx[:, 0].cpu(), Tx[:, 1].cpu(), alpha=0.3, label="T(mu)")
    plt.legend()
    plt.title("Transport Result")
    plt.show()


# ======================
# Main
# ======================
if __name__ == "__main__":
    T = train()
    visualize(T)