import math
import matplotlib.pyplot as plt
import torch 
import numpy as np

def cost(x, y):
    return ((x - y) ** 2).sum(dim=1, keepdim=True)/2

def show_mapping(T,sample_mu,sample_nu, option = False, device = None, f = None, contour = False, legend = True):
    if device is None:
        device = next(T.parameters()).device

    x = sample_mu(1000).to(device)

    outputs = []
    for _ in range(2):
        outputs.append(T(x).detach().cpu())

    outputs = torch.cat(outputs, dim=0)
    x = x.cpu()
    y = sample_nu(1000).cpu()

    plt.figure()

    if (option == True):
        for i in range(50):
            plt.arrow(x[i,0], x[i,1],
                outputs[i,0]-x[i,0],
                outputs[i,1]-x[i,1],
                color='gray', alpha=0.5, head_width=0, length_includes_head=True)
        '''
        for i in range(50):
            plt.arrow(x[i,0], x[i,1],
                outputs[i+n_eval,0]-x[i,0],
                outputs[i+n_eval,1]-x[i,1],
                color='gray', alpha=0.5, head_width=0, length_includes_head=True)
        '''

    if (contour == True and f is not None):
        x_min, x_max = -2.0, 2.0
        y_min, y_max = -2.0, 2.0
        n = 300 

        xs = np.linspace(x_min, x_max, n)
        ys = np.linspace(y_min, y_max, n)
        xx, yy = np.meshgrid(xs, ys)

        grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
        grid_torch = torch.tensor(grid, dtype=torch.float32).to(device)

        with torch.no_grad():
            zz = f(grid_torch).cpu().numpy()
        zz = zz.reshape(n, n)
        plt.contour(xx, yy, zz, levels=20)
        plt.gca().set_aspect("equal")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

    plt.scatter(x[:,0], x[:,1], label=r"$\mu$", s=1, alpha=0.5)
    plt.scatter(y[:,0], y[:,1], label=r"$\nu$", s=1, alpha=0.5)
    plt.scatter(outputs[:,0], outputs[:,1], label=r"$T(x,z)$", s=1, alpha=0.5)
    plt.axis('equal')
    if legend: 
        plt.legend()
    plt.show()

def grad_norm(grads):
    return torch.sqrt(sum((g**2).sum() for g in grads))

def cosine_lr(step, total_steps, lr_max, lr_min=0.0):
    return lr_min + 0.5 * (lr_max - lr_min) * (
        1 + math.cos(math.pi * step / (step+total_steps))
    )

def sigma_schedule(k, K, P=2000, sigma_max=0.2, sigma_min=0.05):
    t = (P * (k // P) + 1) / K
    sigma_k = (1 - t) * sigma_max + t * sigma_min
    return sigma_k

def plot_3d_function(f, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), resolution=200, device=None, cmap="viridis"):
    if device is None:
        try:
            device = next(f.parameters()).device
        except:
            device = torch.device("cpu")

    x_min, x_max = x_range
    y_min, y_max = y_range

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    XY = np.stack([X.ravel(), Y.ravel()], axis=1)
    XY_torch = torch.tensor(XY, dtype=torch.float32, device=device)

    with torch.no_grad():
        Z = f(XY_torch).detach().cpu().numpy()

    Z = Z.reshape(X.shape)

    # plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        X, Y, Z,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.9
    )

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')

    plt.show()