import json
import warnings

from analyze.analyze import analyze_stability_for_molecules
from models import get_model_my_pos_only
from models_edm import get_model
from sampling import sample_prior_pos, sample_flow_pos
from sampling_edm import node2edge_mask
from data.aromatic_dataloader import create_data_loaders

from args import Args
from utils.plotting import plot_graph_of_rings_list

warnings.simplefilter(action='ignore', category=FutureWarning)

import torch

def interp_and_save(args, model, prior, nodes_dist, n_samples=10):
    seed = 5
    n_nodes = 11
    step_size = 1e-1*0

    torch.manual_seed(seed)
    if edm:
        node_mask = torch.ones(args.batch_size, n_nodes, 1).to(args.device)
        z = model.sample_combined_position_feature_noise(
            args.batch_size, n_nodes, node_mask, std=0.7
        )
    else:
        z, node_mask = sample_prior_pos(args, prior, 1, n_nodes, std=0.7)
    edge_mask = node2edge_mask(node_mask[:,:,0])
    edge_mask = edge_mask.view(args.batch_size * args.max_nodes * args.max_nodes, 1).to(args.device)

    z = z.to(args.device)
    step = torch.randn_like(z)
    step -= step.mean(1, keepdim=True)
    step = step / step.norm() * step_size

    molecule_list = []
    for i in range(n_samples):
        if edm:
            x, h = model.sample_from_z(z, node_mask, edge_mask, context=None, fix_noise=False)
            one_hot = h['categorical']
        else:
            x, one_hot = sample_flow_pos(args, model, z, node_mask, 1, n_nodes)
        atom_type = one_hot.argmax(2).cpu().detach()
        x = x.cpu().detach()
        molecule_list += [(x[0], atom_type[0])]
        z += step
    print(f"Generated {n_samples} molecules")

    stability_dict, molecule_stable_list = \
        analyze_stability_for_molecules(molecule_list, args.tol)
    print(f'Stability for {args.exp_dir}, tolerance {args.tol:.1%}')
    for key, value in stability_dict.items():
        try:
            print(f'   {key}: {value:.2%}')
        except:
            pass

    plot_graph_of_rings_list(molecule_list, filename=f'tests/inerp-{step_size:.2f}_{seed}.png',
                                tol=args.tol, title=f'inerp-{step_size:.2f}')

def main(args, edm):
    # Prepare data
    train_loader, val_loader, test_loader = create_data_loaders(args)
    args.context_node_nf = 0  # We can change later if we want condion generation

    # Choose model
    # prior, model, nodes_dist = get_model_my(args, train_loader.dataset.num_node_features)
    if edm:
        model, nodes_dist = get_model(args, train_loader)
        prior = None
    else:
        prior, model, nodes_dist = get_model_my_pos_only(args)

    # Analyze stability, validity, uniqueness and novelty
    with torch.no_grad():
        interp_and_save(
            args, model, prior, nodes_dist
        )

        # # Evaluate negative log-likelihood for the validation and test partitions
        # val_epoch('val', prior, model, nodes_dist, val_loader, args)
        # val_epoch('test', prior, model, nodes_dist, test_loader, args)

if __name__ == '__main__':
    edm = True
    args = Args().parse_args()
    # args.name = 'pos_lr_1e-3_reg_1e-2_1000'
    args.name = 'EDM_9_128_range4'
    args.exp_dir = f'{args.save_dir}/{args.name}'
    print(args.exp_dir)

    with open(args.exp_dir + '/args.txt', "r") as f:
        args.__dict__ = json.load(f)
    args.restore = True
    args.tol = 0.1
    # Automatically choose GPU if available
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("\n\nArgs:", args)

    # Where the magic is
    main(args, edm)
