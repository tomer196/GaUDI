import numpy as np
import matplotlib.pyplot as plt
from torch import logspace, randn_like

from data.aromatic_dataloader import create_data_loaders
from prediction.prediction_args import PredictionArgs
from utils.plotting import plot_graph_of_rings, plot_grap_of_rings_inner


def plot_compare():
    label2dir = {
        "0": "test",
        "1e-3": "test_1e-3",
        "1e-2": "test_1e-2",
        "1e-1": "test_1e-1",
        "1e-0.5": "test_1e-0.5",
    }
    label2losses = {}
    x = np.logspace(-3, -0.5, 10)
    for label, dir in label2dir.items():
        noise_val = np.load(
            f"/home/tomerweiss/PBHs-design/prediction_summary/{dir}/nosie_val.npy"
        )
        label2losses[label] = noise_val
        plt.plot(x[4:], noise_val[4:], label=label)

    plt.legend(title="Noise in training")
    plt.xscale("log")
    plt.xlabel("Noise std")
    plt.ylabel("Val loss")
    plt.show()

    for label, dir in label2dir.items():
        plt.scatter(x[0], label2losses[label][0], label=label)

    plt.legend(title="Noise in training")
    plt.xlabel("Without noise")
    plt.ylabel("Val loss")
    plt.xticks([])
    plt.show()


def vis_noise_mol():
    args = PredictionArgs().parse_args()
    args.augmentation = False
    train_loader, _, _ = create_data_loaders(args)
    x, node_mask, edge_mask, node_features, y = train_loader.dataset[11367]
    x = x[node_mask.bool()]
    # plot_graph_of_rings(x, None, showPlot=True, filename='4r.png')

    # label2std = {
    #     '0': 0,
    #     '1e-3': 1e-3,
    #     '1e-2': 1e-2,
    #     '1e-1': 1e-1,
    #     '1e-0.5': 10**(-0.5),
    # }
    # fig, axes = plt.subplots(1, len(noise_stds), figsize=(30, 30*len(noise_stds)))
    # for i, (label, std) in enumerate(label2std.items()):
    #     plot_grap_of_rings_inner(axes[i], x + randn_like(x)*std, None, tol=0.,
    #                              title=label, align=False)
    # fig.savefig('11234.png', bbox_inches='tight', pad_inches=0.0)


if __name__ == "__main__":
    # plot_compare()
    vis_noise_mol()
    print("Done")
