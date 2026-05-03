import math
import matplotlib.pyplot as plt
import torch 
import numpy as np

def cost(x, y):
    return ((x - y) ** 2).sum(dim=1, keepdim=True)/2

def show_mapping(T,sample_mu,sample_nu, option = False, device = None):
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
        
    plt.scatter(x[:,0], x[:,1], label=r"$\mu$", s=1, alpha=0.5)
    plt.scatter(y[:,0], y[:,1], label=r"$\nu$", s=1, alpha=0.5)
    plt.scatter(outputs[:,0], outputs[:,1], label=r"$T(x,z)$", s=1, alpha=0.5)
    plt.axis('equal')
    plt.legend()
    plt.show()

def grad_norm(grads):
    return torch.sqrt(sum((g**2).sum() for g in grads))

def cosine_lr(step, total_steps, lr_max, lr_min=0.0):
    return lr_min + 0.5 * (lr_max - lr_min) * (
        1 + math.cos(math.pi * step / (step+total_steps))
    )

def plot_3d_function(f, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), resolution=200, device=None, cmap="viridis"):
    if device is None:
        try:
            device = next(f.parameters()).device
        except:
            device = torch.device("cpu")

    x_min, x_max = x_range
    y_min, y_max = y_range

    # grid 생성
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    XY = np.stack([X.ravel(), Y.ravel()], axis=1)
    XY_torch = torch.tensor(XY, dtype=torch.float32, device=device)

    # 함수 평가
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