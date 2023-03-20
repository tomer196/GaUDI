import json
import warnings

from analyze.analyze import analyze_stability_for_molecules
from models_edm import get_model
from sampling_edm import node2edge_mask
from data.aromatic_dataloader import create_data_loaders

from utils.args_edm import Args_EDM
from utils.plotting import plot_graph_of_rings_list, plot_chain

warnings.simplefilter(action="ignore", category=FutureWarning)

import torch


def interp_and_save(args, model):
    seed = 0
    n_nodes = 10
    n_frames = 6
    batch_size = 1
    n_samples = 2

    torch.manual_seed(seed)
    node_mask = torch.ones(batch_size, n_nodes, 1)
    edge_mask = node2edge_mask(node_mask[:, :, 0])

    if args.orientation:
        node_mask = torch.cat([node_mask, node_mask], dim=1)
        edge_mask = torch.cat(
            [
                torch.cat(
                    [
                        edge_mask,
                        torch.eye(n_nodes).unsqueeze(0).repeat(batch_size, 1, 1),
                    ],
                    dim=1,
                ),
                torch.cat([torch.eye(n_nodes), torch.zeros(n_nodes, n_nodes)], dim=0)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1),
            ],
            dim=2,
        )
    edge_mask = edge_mask.view(-1, 1).to(args.device)
    node_mask = node_mask.to(args.device)

    torch.manual_seed(0)
    z = model.sample_combined_position_feature_noise(
        batch_size, node_mask.shape[1], node_mask, std=0.7
    )
    z = z.to(args.device)

    for i in range(n_samples):
        zt = model.sample_chain_from_z(
            z, node_mask, edge_mask, context=None, keep_frames=n_frames - 1
        )
        zt = torch.cat([zt, z])
        print(zt.size())
        n_mask = node_mask[0, :, 0].bool()
        xs = zt[:, n_mask, :3]
        one_hot = zt[:, n_mask, 3:]
        plot_chain(
            xs,
            atom_type=one_hot.argmax(2),
            dirname=f"visualization_scripts/{i}/",
            filename=f"diffusion_{i}",
            dataset=args.dataset,
            orientation=args.orientation,
            gif=False,
            axis_lim=None,
            colors=True,
        )


def main(args):
    # Prepare data
    train_loader, val_loader, test_loader = create_data_loaders(args)

    # Choose model
    model, _, _ = get_model(args, train_loader)

    # Analyze stability, validity, uniqueness and novelty
    with torch.no_grad():
        interp_and_save(args, model)


if __name__ == "__main__":
    args = Args_EDM().parse_args()
    args.name = "EDM_6_192_range4"
    args.name = "hetro_l9_c196_orientation2"
    # args.name = 'cond/lumo_gap_erel_ip_ea__final_act_tanh'
    args.exp_dir = f"{args.save_dir}/{args.name}"
    print(args.exp_dir)

    with open(args.exp_dir + "/args.txt", "r") as f:
        args.__dict__ = json.load(f)
    args.restore = True
    args.tol = 0.1
    if not hasattr(args, "coords_range"):
        args.coords_range = 15
    if not hasattr(args, "conditioning"):
        args.conditioning = False
    # Automatically choose GPU if available
    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    print("Args:", args)

    # Where the magic is
    main(args)
