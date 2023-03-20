# Guided Diffusion for molecular design

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

## Usage
### Training
The training script will train the model using the train and validation datasets. 
When training finish will run evaluation on the test set and will print the results. 
The saved model and the some plots will be save in `summary/{exp_name}`. 
```
python train.py --name 'exp_name' --target_features 'GAP_eV, Erel_eV' 
```

target_features should be separate with `,`.  
Full list of possible arguments can be seen in `utiles/args.py`.  

### Evaluation
Run only the evaluation on trained model.
```
python eval.py --name 'exp_name'
```

### Interpretability
Running the interpretability algorithm. Will save all the molecules in the dataset 
with their GradRAM weights in the same directory as the logs and models.
```
python interpretability_save_all.py --name 'exp_name'
```
To run only on subset of molecules run `interpretability_from_names.py` and change the molecules name
list in line 27.

## Repo structure
- `data` -  Dataset and data related code.
- `se3_trnasformers` - SE3 model use as a predictor
- `utils` - helper function. 
- `summary` - Logs, trained models and interpretability figures for each experiment. 


