# Neural Optimal Transport without Spurious Global Solution

This is the `python` implementation of the paper
**Minimax Optimization without Spurious Solutions in Optimal Transport Learning**.

The repository contains reproducible `ipynb` files that compute optimal transport plans for the weak OT problem without regularization or smoothing techniques.

Examples are provided for reported toy failure cases where spurious global solution exists and for the image translation tasks from `mnist` to `cmnist` Datasets. 

## Setup

To run the notebooks, it is recommended to install *Python 3.12*.
All notebooks are also compatible with `Google Colab` and can be executed directly there.
Otherwise, install the required dependencies as follows:

```
> pip install -r requirements.txt
> python -m pip install torch
```

For `torch` and `torchvision`, installation depends on your system configuration and CUDA version. 
Please refer to the official [PyTorch website](https://pytorch.org/) for detailed instructions.

## Repository Structure



## Train
python train.py

## Reproduce Results
bash scripts/run_all.sh