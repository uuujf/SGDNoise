## CIFAR-10 experiments

This code generates Figure 1 and Figure 6 in the [paper](https://arxiv.org/abs/1803.00195).

### Requirements
1. Python 3.6
2. PyTorch 1.0.0

### Usage
#### Generate data
`python cifar2np.py`

#### Test the performance of compared gradient dynamic
- GD: `python gd.py`
- GLD const: `python gld.py`
- GLD dynamic: `python gld_dyn.py`
- GLD diag: `python gld_diag.py`
- SGD: `python sgd.py`

#### Test the flatness of the minima
`python flatness.py`

### Hyperparameters
See the [paper](https://arxiv.org/abs/1803.00195)