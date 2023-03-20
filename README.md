# Guided Diffusion for Molecular Design

## Usage
First we need to train the unconditioned diffusion model and the time conditioned 
prediction model.
### Training
1. set the required configuration in `utils/args_edm.py` and run:
```python train_edm.py```
2. set the required configuration in `cond_prediction/prediction_args.py` and run:
```python cond_prediction/train_cond_predictor.py```

### Validity evaluation
We can evaluate the stability of the diffusion model by unconditional molecules generation
by updating the experiment name we want to evaluate in line 128 and run:
```python eval_validity.py```


### Conditional generation using guided diffusion
Finally we can start design molecules using the diffusion model. 
1. Configure the paths to the diffusion model and the time conditioned prediction model 
in `generation_guidance.py` lines 225 and 233.
2. Define a target function.
3. Set the gradient scale and number of desired molecules
4. Run:
```python generation_guidance.py```

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
