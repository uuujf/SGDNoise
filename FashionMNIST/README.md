## FashionMNIST experiments

This code generates Figure 3 in the [paper](https://arxiv.org/abs/1803.00195).

### Requirements
1. Python 3.6
2. TensorFlow 1.5.0

### Usage
#### Generate data
`python mnist2np.py`

#### Test the performance of compared gradient dynamic
- GD: `python gd.py`
- GLD const: `python gld.py`
- GLD dynamic: `python gld_dyn.py`
- GLD diag: `python gld_diag.py`
- GLD leading: `python gld_lead.py`
- GLD Hessian: `python gld_hess.py`
- GLD 1st eigvec(H): `python gld_1Hess.py`
- SGD: `python sgd.py`

#### Test the flatness of the minima
`python flatness.py`

### Hyperparameters
See the [paper](https://arxiv.org/abs/1803.00195)