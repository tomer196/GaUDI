# Guided Diffusion for Inverse Molecular Design

The holy grail of material science is \emph{de novo} molecular design -- i.e., the ability to engineer molecules with desired characteristics and functionalities. In recent years, we have witnessed advances in two directions: first, in the ability to predict molecular properties using models such as equivariant graph neural networks; second, in the performance of generation tasks and especially in conditional generation, such as text-to-image generators and large language models. Herein, we combine these two achievements to introduce a guided diffusion model for inverse molecular design, a generative model that enables design of novel molecules with desired properties. We use newly-reported data sets of polycyclic aromatic systems to illustrate the method's effectiveness for a variety of single- and multiple-objective tasks. We demonstrate that the method is capable of generating new molecules with the desired properties and, in some cases, even discovering molecules that are better than the molecules present in our data set of 500,000 molecules.

![GUDI workflow](GaUDI.png)

## Setup
1. Clone this repository by invoking
```
git clone https://github.com/tomer196/GaUDI.git
```
2. Download datasets (`csv` + `xyz`s) from:  
  a. cc-PBH dataset from [COMPAS](https://gitlab.com/porannegroup/compas).  
  b. PASs dataset from [link](https://zenodo.org/record/7798697#.ZCwls-zP1hE).  
3. Update `csv` + `xyz`s paths in `get_paths@data/aromatic_dataloaders.py`.
4. Install conda environment. The environment can be installed using the `environment.yml` by invoking
```
conda env create -n GaUDI --file environment.yml
```
Alternatively, dependencies can be installed manually as follows:
```
conda create -n <env_name> python=3.8
conda activate <env_name>
conda install pytorch=1.10 rdkit
pip install matplotlib networkx tensorboard pandas scipy
```

## Usage
First, we need to train the unconditioned diffusion model (EDM) and the time-conditioned 
property prediction model.

### Training
1. To train the EDM, set the required configuration in `utils/args_edm.py` and run:
```
python train_edm.py
```
This will take 4 hours to train on a single GPU for the cc-PBH dataset. The logs and trained model will be saved in `<save_dir>/<name>`.  

2. To train the predictor, set the required configuration in `cond_prediction/prediction_args.py` and run:
```
python cond_prediction/train_cond_predictor.py
```
The logs and trained model will be saved in `<save_dir>/<name>`. 

### Validity evaluation
To evaluate stability of the unconditional diffusion model,
updatie the experiment name in line 128 of `eval_validity.py` and run:
```
python eval_validity.py
```
Results should be as the results in Table 1 in the paper.


### Conditional generation using guided diffusion
Finally, we can start designing molecules using the guided diffusion model. In order to sample from the guided model, follow these steps:

1. Configure the paths to the diffusion model and the time conditioned prediction model 
in `generation_guidance.py` lines 225 and 233.
2. Define a target function.
4. Set the gradient scale and number of desired molecules
5. Run:
```
python generation_guidance.py
```
When finished, a summary of the results will be printed to `stdout` and the 5 best generated molecules will be saved in `<save_dir>/<name>`. 

Testd on Ubuntu 20.04 with the following libarys varsions:
`pytorch=1.10, matplotlib=3.7.1, networkx=3.0, tensorboard=2.9.1, pandas=1.4.1, scipy=1.10.1`
