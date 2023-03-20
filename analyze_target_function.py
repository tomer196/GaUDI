import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from models_edm import get_model
from prediction.prediction_args import PredictionArgs
from utils.args_edm import Args_EDM
from data.aromatic_dataloader import create_data_loaders

dataset = "cata"
if dataset == "hetro":
    target_features = "HOMO-LUMO gap/eV,HOMO,electron_affinity e/V,ionization e/V"
    df = pd.read_csv(
        "/home/tomerweiss/PBHs-design/data/db-474K-filtered.csv",
        usecols=["name", "nRings"] + target_features.split(","),
    )
    df.rename(columns={"nRings": "n_rings", "name": "molecule"}, inplace=True)
else:
    target_features = "LUMO_eV,GAP_eV,Erel_eV,aIP_eV,aEA_eV"
    # target_features = "HOMO-LUMO gap/eV"
    df = pd.read_csv("/home/tomerweiss/PBHs-design/data/COMPAS-1x.csv")

args = Args_EDM().parse_args()
args.device = "cpu"
args.dataset = dataset
args.orientation = dataset == "hetro"
args.target_features = target_features
train_loader, _, _ = create_data_loaders(args)
_, _, prop_dist = get_model(args, train_loader)


print(df.columns)
df = df[df.n_rings >= 11]
data = torch.Tensor(df[target_features.split(",")].values)


target = torch.Tensor([-8.141, 1.23, 0.606, 11.254, -6.78])[None, :]
# target = torch.Tensor([1.47, -9, 0, 0])[None, :]
print(f"{target=}")
target = prop_dist.normalize(target)
data = prop_dist.normalize(data)


def score_function(values):
    weight = torch.Tensor([[1, 1, 0, 1, 1]]).to(args.device)
    mse = weight * (values - target) ** 2
    erel = values[:, 2]
    return mse.mean(1) + erel


scores = score_function(data)
print(f"Mean score: {scores.mean():.3f}, min: {scores.min():.3f}")
ids = scores.argsort()[:10]
print(scores[ids])
print(prop_dist.unnormalize(data)[ids])
# print(data[ids])
# print(df.iloc[ids].lalas.values)
print(df.iloc[ids].molecule.values)
for t in [1, 2, 2.5, 3, 4]:
    print(
        f"<{t} - {(scores < t).sum()}/{scores.shape[0]} - {((scores < t).sum() / scores.shape[0]).item() * 100:.3f}%"
    )
# print(prop_dist.unnormalize(data[ids[0]]))

np.save(f"scores.npy", scores.numpy().squeeze())
plt.hist(scores.numpy().squeeze(), density=True)
# plt.hist(dist.numpy().squeeze(), bins=torch.linspace(-3, 3, 20), density=True)
# plt.xlim([0,10])
# plt.ylim([0,0.4])
# plt.show()
# plt.savefig(f'scores_hist_data.png', bbox_inches='tight', pad_inches=0.0)
