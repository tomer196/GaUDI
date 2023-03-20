import json
import os
import random
from datetime import datetime
from time import time
import warnings
from typing import Tuple

import numpy as np
from torch import Tensor, randn
import matplotlib.pyplot as plt

from analyze.analyze import analyze_stability_for_molecules
from edm.equivariant_diffusion.utils import (
    assert_correctly_masked,
    assert_mean_zero_with_mask,
)
from models_edm import get_model
from prediction.prediction_args import PredictionArgs
from prediction.train_cond_predictor import get_cond_predictor_model
from sampling_edm import sample_guidance
from data.aromatic_dataloader import create_data_loaders

from utils.args_edm import Args_EDM
from utils.plotting import plot_graph_of_rings, plot_rdkit

warnings.simplefilter(action="ignore", category=FutureWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
scale = 100
n_nodes = 10

import torch


@torch.no_grad()
def predict(model, x, h, node_mask, edge_mask, edm_model) -> Tuple[Tensor, Tensor]:
    bs, n_nodes, n_dims = x.size()
    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
    node_mask = node_mask.view(bs, n_nodes, 1)

    t = torch.zeros(x.size(0), 1, device=x.device).float()
    x, h, _ = edm_model.normalize(
        x,
        {"categorical": h, "integer": torch.zeros(0).to(x.device)},
        node_mask,
    )
    xh = torch.cat([x, h["categorical"]], dim=-1)
    pred = model(xh, node_mask, edge_mask, t)
    return pred


@torch.no_grad()
def get_score(x, h, score_function, node_mask, edge_mask, edm_model):
    bs, n_nodes, n_dims = x.size()
    # edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
    # edge_mask = edge_mask.repeat(bs, 1, 1).view(-1, 1).to(args.device)
    # node_mask = torch.ones(bs, n_nodes, 1).to(args.device)
    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
    node_mask = node_mask.view(bs, n_nodes, 1)

    t = torch.zeros(x.size(0), 1, device=x.device).float()
    x, h, _ = edm_model.normalize(
        x,
        {"categorical": h, "integer": torch.zeros(0).to(x.device)},
        node_mask,
    )
    xh = torch.cat([x, h["categorical"]], dim=-1)
    return score_function(xh, node_mask, edge_mask, t)


def eval_stability(x, one_hot, dataset="cata", orientation=False):
    atom_type = one_hot.argmax(2).cpu().detach()
    molecule_list = [(x[i].cpu().detach(), atom_type[i]) for i in range(x.shape[0])]
    stability_dict, molecule_stable_list = analyze_stability_for_molecules(
        molecule_list, args.tol, dataset=dataset, orientation=orientation
    )
    x = x[stability_dict["molecule_stable_bool"]]
    atom_type = atom_type[stability_dict["molecule_stable_bool"]]
    return stability_dict, x, atom_type


def sample_z(mu, sigma, n_samples):
    z = randn(n_samples, mu.shape[0], device=mu.device)
    z = mu + torch.einsum("ij,bj->bi", sigma, z)
    z = z.contiguous().view(n_samples, -1, 3)
    z = z - z.mean(1, keepdim=True)
    return z


def switch_grad_off(models):
    for m in models:
        for p in m.parameters():
            p.requires_grad = False


def design(args, model, cond_predictor, nodes_dist, prop_dist, scale, n_nodes):
    switch_grad_off([model, cond_predictor])
    model.eval()
    cond_predictor.eval()
    print("Design molecule...")
    nodesxsample = Tensor([n_nodes] * args.batch_size).long()
    # nodesxsample = nodes_dist.sample(args.batch_size)
    # nodesxsample = torch.arange(5,12).repeat(100)

    c_steps = 1
    target = torch.Tensor([-8.141, 1.23, 0.5, 11.254, -6.78])
    # target = torch.Tensor([1.47, -9, 0, 0, 0])
    # target = torch.Tensor([-8.141, 1.23, 0.5, 11.254, -6.278])
    # target = None
    # target = 4
    if target is not None:
        context = target[None, :].repeat(nodesxsample.shape[0], 1)
        context = prop_dist.normalize(context)
    else:
        # context = prop_dist.sample_df(nodesxsample)
        context = prop_dist.sample_batch(nodesxsample)
        # print(prop_dist.unnormalize(context))
    context = context.to(args.device)

    # def target_function(_input, _node_mask, _edge_mask, _t):
    #     pred = cond_predictor(_input, _node_mask, _edge_mask, _t)
    #     weight = Tensor([[1, 1, 0, 1, 1]]).to(args.device)
    #     mse = weight * (pred - context) ** 2
    #     erel = pred[:, 2]
    #     return mse.mean(1) + erel

    def target_function(_input, _node_mask, _edge_mask, _t):
        pred = cond_predictor(_input, _node_mask, _edge_mask, _t)
        pred = prop_dist.unnormalize(pred)
        gap = pred[:, 0]
        ea = pred[:, 2] * 27.2114
        ip = pred[:, 3] * 27.2114
        return ip + ea + 3 * gap
        # return -gap

    start_time = time()
    x, one_hot, node_mask, edge_mask = sample_guidance(
        args,
        model,
        target_function,
        nodesxsample,
        context=None,
        c_steps=c_steps,
        c_scale=scale,
    )
    print(f"Generated {x.shape[0]} molecules in {time() - start_time:.2f} seconds")
    assert_correctly_masked(x, node_mask)
    assert_mean_zero_with_mask(x, node_mask)

    stability_dict, x_stable, atom_type_stable = eval_stability(
        x, one_hot, dataset=args.dataset, orientation=args.orientation
    )
    print(f"{scale=}, {c_steps=}, {target}")
    print(f"{stability_dict['mol_stable']=:.2%} out of {x.shape[0]}")
    scores = (
        get_score(x, one_hot, target_function, node_mask, edge_mask, model)
        .detach()
        .cpu()
    )
    pred = (
        predict(cond_predictor, x, one_hot, node_mask, edge_mask, model).detach().cpu()
    )

    pred = prop_dist.unnormalize(pred)
    # pred[:, 2:4] *= 27.2114
    context = prop_dist.unnormalize(context.cpu())
    print(f"Mean score: {scores.mean().item():.4f}")

    timestamp = datetime.now().strftime("%m%d_%H:%M:%S")
    dir_name = f"best/{timestamp}_{scale}"
    os.mkdir(dir_name)

    best_idx = scores.min(0).indices.item()
    best_value = scores[best_idx]
    atom_type = one_hot.argmax(2).cpu().detach()
    print(f"best value: {pred[best_idx]}, " f"score: {best_value}")
    best_str = ", ".join([f"{t:.3f}" for t in pred[best_idx]])
    plot_graph_of_rings(
        x[best_idx].detach().cpu(),
        atom_type[best_idx].detach().cpu(),
        filename=f"{dir_name}/all.png",
        tol=args.tol,
        title=f"{best_str}\n {best_value}",
        dataset=args.dataset,
    )

    pred = pred[stability_dict["molecule_stable_bool"]]
    # context = context[stability_dict['molecule_stable_bool']]
    scores = scores[stability_dict["molecule_stable_bool"]]
    print(f"Mean score: {scores.mean().item():.4f}")

    best_idxs = scores.argsort()
    for i in range(min(5, scores.shape[0])):
        idx = best_idxs[i]
        value = scores[idx]
        print(f"best value (from stable): {pred[idx]}, " f"score: {value}")

        # target_str = ", ".join([f"{t:.3f}" for t in target])
        best_str = ", ".join([f"{t:.3f}" for t in pred[idx]])
        plot_graph_of_rings(
            x_stable[idx].detach().cpu(),
            atom_type_stable[idx].detach().cpu(),
            filename=f"{dir_name}/{i}.pdf",
            tol=args.tol,
            title=f"{best_str}\n {value}",
            dataset=args.dataset,
        )
        plot_rdkit(
            x_stable[idx].detach().cpu(),
            atom_type_stable[idx].detach().cpu(),
            filename=f"{dir_name}/mol_{i}.pdf",
            tol=args.tol,
            title=f"{best_str}\n {value}",
            dataset=args.dataset,
        )

    np.save(f"{dir_name}/scores_{scale}.npy", scores.numpy().squeeze())
    # plt.hist(scores.numpy().squeeze(), bins=torch.linspace(-3, 3, 20), density=True)
    plt.close()
    plt.hist(scores.numpy().squeeze(), density=True)
    # plt.xlim([0,10])
    # plt.ylim([0,0.4])
    plt.show()
    plt.savefig(f"scores_hist_{scale}.png", bbox_inches="tight", pad_inches=0.0)
    for t in [1, 2, 2.5, 3, 4]:
        print(
            f"<{t} - {(scores < t).sum()}/{scores.shape[0]} - "
            f"{((scores < t).sum() / scores.shape[0]).item() * 100:.3f}%"
        )


def main(args, cond_predictor_args):
    args.batch_size = 512
    args.tol = 0.1

    train_loader, _, _ = create_data_loaders(cond_predictor_args)
    model, nodes_dist, prop_dist = get_model(args, train_loader)
    cond_predictor = get_cond_predictor_model(cond_predictor_args, train_loader.dataset)

    design(args, model, cond_predictor, nodes_dist, prop_dist, scale, n_nodes)


if __name__ == "__main__":
    args = Args_EDM().parse_args()
    # args.name = "cond/lumo_gap_erel_ip_ea"
    # args.name = 'cond/lumo_gap_erel_ip_ea__final_act_tanh'
    # args.name = "cond/lumo_gap_erel_ip_ea_uncond"
    args.name = "hetro_l9_c196_orientation2"
    # args.name = "cata_l9_c196_polynomial_2"
    args.exp_dir = f"{args.save_dir}/{args.name}"
    print(args.exp_dir)

    with open(args.exp_dir + "/args.txt", "r") as f:
        args.__dict__ = json.load(f)
    args.restore = True
    if not hasattr(args, "conditioning"):
        args.conditioning = False

    cond_predictor_args = PredictionArgs().parse_args()
    cond_predictor_args.name = "corrector/lumo_gap_erel_ip_ea"
    # cond_predictor_args.name = (
    #     "cond_predictor/hetro_gap_l12_c196_6e-4_orientation2_range4_dp"  # model750
    # )
    cond_predictor_args.name = (
        "cond_predictor/hetro_gap_homo_ea_ip_stability_polynomial_2_with_norm"
    )
    # cond_predictor_args.name = (
    #     "cond_predictor/cata_lumo_gap_erel_ip_ea_polynomial_2_with_norm"
    # )
    cond_predictor_args.exp_dir = (
        f"{cond_predictor_args.save_dir}/{cond_predictor_args.name}"
    )
    print(cond_predictor_args.exp_dir)

    with open(cond_predictor_args.exp_dir + "/args.txt", "r") as f:
        cond_predictor_args.__dict__ = json.load(f)
    cond_predictor_args.restore = True
    # cond_predictor_args.coords_range = 15

    # Automatically choose GPU if available
    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    cond_predictor_args.device = args.device

    # Where the magic is
    main(args, cond_predictor_args)
