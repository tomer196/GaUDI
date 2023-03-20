import json
import math
import random
import warnings

from analyze.analyze import analyze_stability_for_molecules
from data.aromatic_dataloader import create_data_loaders
from data.gor2goa import analyze_rdkit_valid_for_molecules
from models_edm import get_model
from sampling_edm import sample_pos_edm, save_and_sample_chain_edm
from utils.args_edm import Args_EDM

from utils.plotting import plot_graph_of_rings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import torch


def analyze_and_save(args, model, nodes_dist, prop_dist, n_samples=1000):
    print("-" * 20)
    print("Generate molecules...")

    n_nodes = 10
    # args.max_nodes = 72
    nodesxsample = torch.Tensor([n_nodes] * args.batch_size).long()
    if args.conditioning:
        context = prop_dist.sample_batch(nodesxsample)
        context = prop_dist.normalize(context)
    else:
        context = None

    molecule_list = []
    n_samples = math.ceil(n_samples / args.batch_size) * args.batch_size
    for i in range(n_samples // args.batch_size):
        n_samples = min(args.batch_size, n_samples)
        # nodesxsample = nodes_dist.sample(n_samples)
        x, one_hot, node_mask, edge_mask = sample_pos_edm(
            args, model, prop_dist, nodesxsample, std=1.0, context=context
        )

        x = x.cpu().detach()
        one_hot = one_hot.cpu().detach()
        x = [x[i][node_mask[i, :, 0].bool()] for i in range(x.shape[0])]
        atom_type = [
            one_hot[i][node_mask[i, :, 0].bool()].argmax(dim=1) for i in range(len(x))
        ]
        molecule_list += [(x[i], atom_type[i]) for i in range(len(x))]

    print(f"{len(molecule_list)} molecules generated, starting analysis")

    stability_dict, molecule_stable_list = analyze_stability_for_molecules(
        molecule_list, args.tol, dataset=args.dataset, orientation=args.orientation
    )
    print(f"Stability for {args.exp_dir}, tolerance {args.tol:.1%}")
    for key, value in stability_dict.items():
        try:
            print(f"   {key}: {value:.2%}")
        except:
            pass

    stability_dict, molecule_stable_list = analyze_rdkit_valid_for_molecules(
        molecule_list, args.tol, dataset=args.dataset
    )
    print(f"RDkit validity for {args.exp_dir}, tolerance {args.tol:.1%}")
    for key, value in stability_dict.items():
        try:
            print(f"   {key}: {value:.2%}")
        except:
            pass

    # plot some molecules
    non_stable_list = list(set(molecule_list) - set(molecule_stable_list))
    if len(non_stable_list) != 0:
        idxs = np.random.randint(0, len(non_stable_list), 5)
        for i in idxs:
            x, atom_type = non_stable_list[i]
            title = f"Non-stable-{i}"
            plot_graph_of_rings(
                x,
                atom_type,
                filename=f"{args.exp_dir}/{title}.png",
                tol=args.tol,
                dataset=args.dataset,
            )
    if len(molecule_stable_list) != 0:
        idxs = np.random.randint(0, len(molecule_stable_list), 5)
        for i in idxs:
            x, atom_type = molecule_stable_list[i]
            title = f"Stable-{i}"
            plot_graph_of_rings(
                x,
                atom_type,
                filename=f"{args.exp_dir}/{title}.png",
                tol=args.tol,
                dataset=args.dataset,
            )

    # create chains
    for i in range(args.n_chains):
        save_and_sample_chain_edm(
            args,
            model,
            prop_dist,
            dirname=args.exp_dir,
            file_name=f"chain{i}",
            std=0.7,
            n_tries=10,
        )
    return stability_dict


def main(args):
    # Prepare data
    train_loader, val_loader, test_loader = create_data_loaders(args)

    model, nodes_dist, prop_dist = get_model(args, train_loader)

    # Analyze stability, validity, uniqueness and novelty
    with torch.no_grad():
        analyze_and_save(args, model, nodes_dist, prop_dist, n_samples=args.n_samples)


if __name__ == "__main__":
    args = Args_EDM().parse_args()

    # args.name = 'pos_lr_1e-3_reg_1e-2_1000'
    # args.name = 'pos_lr_1e-3_reg_1e-2_save'
    args.name = "EDM_6_192_range4"
    args.name = "cond/lumo_gap_erel_ip_ea_uncond"
    # args.name = 'test-hetro'
    args.name = "hetro_l9_c196_orientation2"
    # args.name = "cata_l9_c196_polynomial_2"
    # args.name = "atoms/cata_l9_c196_polynomial_2"
    args.exp_dir = f"{args.save_dir}/{args.name}"
    print(args.exp_dir)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    with open(args.exp_dir + "/args.txt", "r") as f:
        args.__dict__ = json.load(f)
    args.restore = True
    args.n_samples = 1000
    # args.batch_size = 256
    args.n_chains = 0
    args.tol = 0.1
    if not hasattr(args, "coords_range"):
        args.coords_range = 15
    if not hasattr(args, "conditioning"):
        args.conditioning = False
    if not hasattr(args, "dataset"):
        args.dataset = "cata"
    # Automatically choose GPU if available
    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    print("Args:", args)

    # Where the magic is
    main(args)
