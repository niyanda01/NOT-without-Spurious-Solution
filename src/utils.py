def cost(x, y):
    return ((x - y) ** 2).sum(dim=1, keepdim=True)/2

def show_mapping(T,sample_mu,sample_nu):
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