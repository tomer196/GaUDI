# Guided Diffusion for Inverse Molecular Design

The holy grail of material science is \emph{de novo} molecular design -- i.e., the ability to engineer molecules with desired characteristics and functionalities. In recent years, we have witnessed advances in two directions: first, in the ability to predict molecular properties using models such as equivariant graph neural networks; second, in the performance of generation tasks and especially in conditional generation, such as text-to-image generators and large language models. Herein, we combine these two achievements to introduce a guided diffusion model for inverse molecular design, a generative model that enables design of novel molecules with desired properties. We use newly-reported data sets of polycyclic aromatic systems to illustrate the method's effectiveness for a variety of single- and multiple-objective tasks. We demonstrate that the method is capable of generating new molecules with the desired properties and, in some cases, even discovering molecules that are better than the molecules present in our data set of 500,000 molecules.

[GUDI workflow](https://github.com/tomer196/GUDI/blob/main/GUDI.png)

## Usage
First we need to train the unconditioned diffusion model and the time conditioned 
prediction model.
### Training
1. set the required configuration in `utils/args_edm.py` and run:
```
python train_edm.py
```
2. set the required configuration in `cond_prediction/prediction_args.py` and run:
```
python cond_prediction/train_cond_predictor.py
```

### Validity evaluation
We can evaluate the stability of the diffusion model by unconditional molecules generation
by updating the experiment name we want to evaluate in line 128 and run:
```
python eval_validity.py
```


### Conditional generation using guided diffusion
Finally we can start design molecules using the diffusion model. 
1. Configure the paths to the diffusion model and the time conditioned prediction model 
in `generation_guidance.py` lines 225 and 233.
2. Define a target function.
3. Set the gradient scale and number of desired molecules
4. Run:
```
python generation_guidance.py
```

## preparations
1. Download repo.  
2. Download dataset (`csv` + `xyz`s) from [COMPAS](https://gitlab.com/porannegroup/compas)
3. Update `csv` + `xyz`s paths in `get_paths@data/aromatic_dataloaders.py`
4. Install conda environment according to the instruction below

## Dependencies
```
conda create -n <env_name> python=3.8
conda activate <env_name>
conda install pytorch=1.10 rdkit
pip install matplotlib networkx tensorboard pandas scipy
```
