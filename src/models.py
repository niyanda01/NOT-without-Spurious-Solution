import torch
import torch.nn as nn
import torch.nn.functional as F

class Transport(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        z = torch.randn(x.shape[0], 1, device=x.device)
        # z = 2 * torch.rand(x.shape[0], 1, device=x.device) - 1
        return self.net(torch.cat([x, z], dim=1))


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

class ICNNCritic(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()

        self.wx1 = nn.Linear(input_dim, hidden_dim)

        self.wz2_raw = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.wx2 = nn.Linear(input_dim, hidden_dim)

        self.wz3_raw = nn.Parameter(torch.randn(1, hidden_dim))
        self.wx3 = nn.Linear(input_dim, 1)

        self.bias2 = nn.Parameter(torch.zeros(hidden_dim))
        self.bias3 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # first layer
        z = F.relu(self.wx1(x))

        # enforce non-negative weights
        wz2 = F.softplus(self.wz2_raw)
        wz3 = F.softplus(self.wz3_raw)

        # second layer
        z = F.relu(F.linear(z, wz2) + self.wx2(x) + self.bias2)

        # output layer (no activation → convex 유지)
        out = F.linear(z, wz3) + self.wx3(x) + self.bias3

        return 0.5 * torch.sum(x**2, dim=1, keepdim=True) - out/10

class RF_Transport(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=500, output_dim=2, scale=0.3):
        super().__init__()

        # random feature (freeze)
        self.register_buffer("W1", torch.randn(input_dim, hidden_dim) * scale)
        self.register_buffer("b1", torch.randn(hidden_dim) * scale)

        # trainable linear head
        self.W2 = nn.Parameter(torch.randn(hidden_dim, output_dim) * scale)
        self.b2 = nn.Parameter(torch.randn(output_dim)* scale)

    def forward(self, z):
        phi = nn.Sigmoid()
        h1 = 2 * phi(z @ self.W1 + self.b1)
        out = h1 @ self.W2 + self.b2
        return out

class RF_Critic(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=1000, scale = 0.3):
        super().__init__()

        # random features (fixed)
        self.register_buffer("W", torch.randn(input_dim, hidden_dim) * scale)
        self.register_buffer("b", torch.randn(hidden_dim) * scale)

        # trainable nonnegative output weight
        self.a = nn.Parameter(torch.rand(hidden_dim, 1))
        self.phi = nn.ReLU()

    def forward(self, x):
        h = self.phi(x @ self.W + self.b)

        out = 0.5 * torch.sum(x**2, dim=1, keepdim=True) - h @ self.a

        return out