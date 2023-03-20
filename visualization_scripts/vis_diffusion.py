import warnings
from models_edm import get_model
from data.aromatic_dataloader import create_data_loaders
import matplotlib.pyplot as plt
from utils.args_edm import Args_EDM
from utils.plotting import plot_graph_of_rings_list, plot_graph_of_rings, plot_chain

warnings.simplefilter(action="ignore", category=FutureWarning)

import torch


def main():
    args = Args_EDM().parse_args()
    args.num_workers = 0
    args.tol = 0.1
    n_frames = 5
    n_samples = 1
    args.device = torch.device("cpu")

    args.batch_size = 1
    # Prepare data
    train_loader, val_loader, _ = create_data_loaders(args)
    train_loader.shuffle = False
    args.context_node_nf = 0
    # Choose model
    model, nodes_dist, prop_dist = get_model(args, train_loader)
    # plt.plot(model.gamma.show_schedule(100))
    # plt.title(args.diffusion_noise_schedule)
    # plt.show()
    x, node_mask, edge_mask, node_features, y = val_loader.dataset[1]
    x = x.unsqueeze(0)
    node_mask = node_mask.unsqueeze(0)
    node_features = node_features.unsqueeze(0)
    for sample in range(n_samples):
        # plot_graph_of_rings(x[0, node_mask[0].bool()], None, filename=f'tests/{sample}.png', tol=args.tol)

        h = {"categorical": node_features, "integer": torch.zeros(0)}
        zt = model.diffusion_chain(x, h, node_mask, keep_frames=n_frames)
        print(zt.size())
        n_mask = node_mask[0].bool()
        xs = zt[:, n_mask, :3]
        one_hot = zt[:, n_mask, 3:]
        plot_chain(
            xs,
            atom_type=one_hot.argmax(2),
            dirname=f"visualization_scripts/{sample}/",
            filename=f"diffusion_{sample}",
            dataset=args.dataset,
            orientation=args.orientation,
            gif=False,
        )


if __name__ == "__main__":
    main()
