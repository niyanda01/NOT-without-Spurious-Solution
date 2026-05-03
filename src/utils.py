def cost(x, y):
    return ((x - y) ** 2).sum(dim=1, keepdim=True)/2

