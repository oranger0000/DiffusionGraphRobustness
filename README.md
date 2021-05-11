# DiffusionGraphRobustness

This repo contains the artifacts for our project **Understanding the Impact of Graph Diffusion on Robust Graph Learning** as part of CS6604: 'Data Challenges in Machine Learning' course.

The project report can be accessed from [here](https://github.com/satvikchekuri/DiffusionGraphRobustness/blob/main/examples/graph/figures/report.pdf).

## Acknowledgment

Graph Diffusion was implemented in our project using models implemented in [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) Python library. 

Attack model and base Graph Neural Networks was implemented in or project using models implemented in [DeepRobust](https://github.com/DSE-MSU/DeepRobust) Python library.
## Requirements
It is advised to create a new environment and install the below packages along with PyTorch.

torch-geometric==1.7.0 (Installation steps [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html))
deeprobust==0.2.1 (Installation steps [here](https://deeprobust.readthedocs.io/en/latest/notes/installation.html))



### Versions of software/libraries/OS used to run experiments on the repo owner's system
CUDA: 10.1

PyTorch: 1.8.1

Python: 3.8

Ubuntu 18.04

## Implemented Models 

<p align="center">
<img center src="https://github.com/satvikchekuri/DiffusionGraphRobustness/blob/main/examples/graph/figures/graph.PNG" width = "650" alt="logo">
</p>

**Attack method: Metattack**

Type: Global attack; Poisoning; Domain: Node Classification; [Paper](https://openreview.net/pdf?id=Bylnx209YX)

**GNN: ChebNet**

Domain: Node Classification; [Paper](https://arxiv.org/abs/1606.09375)

**GNN: SGC**

Domain: Node Classification; [Paper](https://arxiv.org/abs/1902.07153)

**GNN: ChebNet**

Domain: Node Classification; [Paper](https://arxiv.org/abs/1710.10903)

**Diffusion: Graph Diffusion Convolution**

[Paper](https://arxiv.org/pdf/1911.05485.pdf)

## Datasets

<p align="center">
<img center src="https://github.com/satvikchekuri/DiffusionGraphRobustness/blob/main/examples/graph/figures/graph1.PNG" width = "450" alt="logo">
</p>

The 'cora', 'cora-ml', 'polblogs' and 'citeseer' are downloaded from [here](https://github.com/danielzuegner/gnn-meta-attack/tree/master/data), and 'pubmed' is from [here](https://github.com/tkipf/gcn/tree/master/gcn/data).

-----------------------------------------------------------------------------------------------------------------------

# Run Experiments

## Pre-steps

To accommodate the change in data format after performing Graph Diffusion, we had to make few additions in the python files related to GNN architectures ChebNet, SGC, GAT in DeepRobust package. 
Please follow the steps below:

1) Make sure you have successfully finished installing DeepRobust library in your system as well as cloned this repo to your system.

2) Copy the following three python files in **/examples/defense/** folder to the DeepRobust package in your system.
    
    chebnet.py | gat.py | sgc.py
    
    Destination folder: **/home/[your system name]/anaconda3/envs/[your env name]/lib/python3.8/site-packages/deeprobust-0.2.0-py3.8/deeprobust/graph/defense**

3) Sample: 
``` bash
$ cp /home/satvik/PyCharmProjects/DiffusionGraphRobustness/examples/defense/chebnet.py /home/satvik/anaconda3/envs/graph/lib/python3.8/site-packages/deeprobust-0.2.0-py3.8/deeprobust/graph/defense/chebnet.py
```

## Experiment

##### To run experiments go to /examples/graph/ folder and run
``` bash
$ bash run_all.sh sgc cora 0.05
```
$1 = base GNN [Options: chebnet, sgc, gat]

$2 = Dataset [Options: cora, cora_ml, citeseer, polblogs, pubmed]

$3 = Perturbation Ratio [ptions: 0.05, 0.1, 0.15, 0.2, 0.25]

The output would be test-loss & accuracy for a clean graph, perturbed graph, and diffused graph with perturbation. 

We faced issues with running diffusion experiments on PubMed dataset due to GPU memory. Please be careful when running on that dataset as the system might freeze.
