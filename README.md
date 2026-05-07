# Neural Optimal Transport without Spurious Global Solution

This is the `python` implementation of the paper
**Minimax Optimization without Spurious Solutions in Optimal Transport Learning**.

The repository contains reproducible `ipynb` files that compute optimal transport plans for the weak OT problem without regularization or smoothing techniques.

Examples are provided for reported toy failure cases where spurious global solution exists and for the image translation tasks from `mnist` to `cmnist` Datasets. 

## Setup

To run the notebooks, it is recommended to install *Python 3.12*.
All notebooks are also compatible with `Google Colab` and can executed directly there when clone source codes.

```
> !git clone https://github.com/niyanda01/NOT-without-Spurious-Solution.git
> %cd NOT-without-Spurious-Solution
```

Otherwise, install the required dependencies as follows:

```
> pip install -r requirements.txt
> python -m pip install torch
```

For `torch` and `torchvision`, installation depends on your system configuration and CUDA version. 
Please refer to the official [PyTorch website](https://pytorch.org/) for detailed instructions.

## Repository Structure

**2D toy experiment**
- `notebook/1. NOT_toy_GDmax.ipynb` - learning perpendicular distribution with GDmax and EG.
- `notebook/2. NOT_toy_ICNN.ipynb` - learning perpendicular distribution with ICNN.
- `notebook/3. NOT_toy_EG.ipynb` -learning various synthetic data with spurious solution with EG. 
- `notebook/4. NOT_toy_Smoothing.ipynb` - learning synthetic data with smoothing.

**CMNIST Image Translation**
- `notebook/5. NOT_CMNIST.ipynb` - learning random color cmnist to cmnist image translation
- `notebook/5. NOT_CMNIST_Orthogonal.ipynb` - learning red color mnist to blue color cmnist image translation