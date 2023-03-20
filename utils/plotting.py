import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem
from torch import Tensor

from data.aromatic_dataloader import ATOMS_LIST, RINGS_LIST
from data.gor2goa import gor2goa, rdkit_valid

# from matplotlib import cm
# import networkx as nx

from utils.utils import positions2adj


def get_figure(_molrepr, _edges, filename=None, showPlot=False):
    """
    get_figure(_molrepr: Mol Object, _egdes: list, filename: string)

    Function that draws the molecule and saves and/or shows it.

    in:
    _molrepr: Mol Object containing XYZ data.
    _edges: list of tuples containing the indices of 2 atoms bonding.
    filename: name of the file where the figure will be saved.

    """

    # set parameters
    plt.rcParams.update({"font.size": 22})

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    # plot scan and molecule with path
    moldraw(ax, _molrepr, _edges)

    # save figure
    if filename:
        fig.savefig(filename)

    # show figure
    if showPlot:
        plt.show()

    return None


def moldraw(ax, _molrepr, _edges, plot_h=False):
    """
    moldraw(ax, _molrepr: Mol Object, _edges: list)

    Function that draws the molecule.

    in:
    _molrepr: Mol Object containing XYZ data.
    _edges: list of tuples containing the indices of 2 atoms bonding.

    """

    atom_colors = {
        "H": "silver",
        "N": "blue",
        "O": "red",
        "S": "goldenrod",
        "B": "green",
    }

    # set aspect of subplot
    ax.set_aspect("equal")
    ax.axis("off")

    # plot molecule
    for edge in _edges:
        bond = []
        Hbond = False
        for atom_idx in edge:
            for atom in _molrepr.atoms:
                if atom_idx == atom.index:
                    bond.append(atom)
                    if atom.element != "C":
                        if atom.element == "BH":
                            atom.element = "B"
                        elif atom.element == "H":
                            Hbond = True
                        if Hbond and not plot_h:
                            continue
                        ax.text(
                            atom.x,
                            atom.y,
                            atom.element,
                            ha="center",
                            va="center",
                            color=atom_colors[atom.element],
                            zorder=2,
                            bbox=dict(
                                facecolor="white",
                                edgecolor="none",
                                boxstyle="circle, pad=0.1",
                            ),
                        )
        x = [atom.x for atom in bond]
        y = [atom.y for atom in bond]
        if Hbond:
            if plot_h:
                ax.plot(x, y, c=atom_colors["H"], linestyle="-", zorder=0)
        else:
            ax.plot(x, y, c="black", linestyle="-", zorder=1)

    return None


def plot_compare(pred, gt, args, title=""):
    # gt = np.concatenate([gt[:1405], gt[1406:]])
    # pred = np.concatenate([pred[:1405], pred[1406:]])
    min_val = np.concatenate([gt, pred]).min()
    max_val = np.concatenate([gt, pred]).max()
    plt.scatter(gt, pred)
    plt.plot([min_val, max_val], [min_val, max_val])
    plt.title(
        f"{title}, MAE: {np.abs(gt-pred).mean():.4f}+-{np.abs(gt-pred).std():.4f}"
    )
    plt.xlabel("Ground truth")
    plt.ylabel("Prediction")
    plt.savefig(args.exp_dir + "/" + title.replace("/", "") + ".png")
    plt.show()


def align_to_xy_plane(x):
    """
    Rotate the molecule into xy-plane.

    """
    I = np.zeros((3, 3))  # set up inertia tensor I
    com = np.zeros(3)  # set up center of mass com

    # calculate moment of inertia tensor I
    for i in range(x.shape[0]):
        atom = x[i]
        I += np.array(
            [
                [(atom[1] ** 2 + atom[2] ** 2), -atom[0] * atom[1], -atom[0] * atom[2]],
                [-atom[0] * atom[1], (atom[0] ** 2 + atom[2] ** 2), -atom[1] * atom[2]],
                [-atom[0] * atom[2], -atom[1] * atom[2], atom[0] ** 2 + atom[1] ** 2],
            ]
        )

        com += atom
    com = com / len(com)

    # extract eigenvalues and eigenvectors for I
    # np.linalg.eigh(I)[0] are eigenValues, [1] are eigenVectors
    eigenVectors = np.linalg.eigh(I)[1]
    eigenVectorsTransposed = np.transpose(eigenVectors)

    a = []
    for i in range(x.shape[0]):
        xyz = x[i]
        a.append(np.dot(eigenVectorsTransposed, xyz - com))
    return np.stack(a)


def align_to_x_plane(x):
    """
    Rotate the molecule into x axis.
    """
    x = x[:, :2]
    xm = x.mean(0)
    xc = x - xm
    _, _, Vt = np.linalg.svd(xc)
    xr = xc @ Vt.T
    return xr, xm, Vt


def plot_graph(g, title="", ax=None):
    if ax is None:
        ax = plt.gca()
    x = g.ndata["x"].detach().cpu().numpy()
    # scatter3d(x[:, 0], x[:, 1], x[:, 2])
    x = align_to_xy_plane(x)
    # scatter(x[:, 0], x[:, 1])
    edges = torch.stack(g.edges(), dim=1).detach().cpu().numpy()
    # remove double edges
    n = int(edges.shape[0] / 2)
    edges = edges[:n]

    # if we want later we can add node/edge type (extract from the features)
    plt.rcParams.update({"font.size": 16})
    ax.set_aspect("equal")
    ax.axis("off")
    ax.scatter(x[:, 0], x[:, 1])
    for edge_ind in range(edges.shape[0]):
        edge = edges[edge_ind]
        ax.plot(x[edge, 0], x[edge, 1], c="black")
    ax.set_title(f"{title}")
    plt.show()
    return ax


def plot_grap_of_rings_inner(
    ax,
    x: Tensor,
    atom_type,
    title="",
    tol=0.1,
    axis_lim=10,
    align=True,
    dataset="cata",
    orientation=False,
    adj=None,
):
    x = torch.clamp(x, min=-1e5, max=1e5)
    rings_list = RINGS_LIST["hetro"]
    if orientation:
        n = x.shape[0] // 2
        if adj is None:
            _, adj = positions2adj(
                x[None, :n, :], atom_type[None, :n], tol=tol, dataset=dataset
            )
            adj = adj[0]
            adj = torch.cat(
                [
                    torch.cat([adj, torch.eye(n)], dim=1),
                    torch.cat([torch.eye(n), torch.zeros(n, n)], dim=1),
                ],
                dim=0,
            )
    elif adj is None:
        _, adj = positions2adj(
            x[None, :, :], atom_type[None, :], tol=tol, dataset=dataset
        )
        adj = adj[0]

    x = x.cpu().numpy()
    if align:
        x = align_to_xy_plane(x)
        x -= x.mean(0)

    ax.scatter(x[:, 0], x[:, 1], c="blue")
    ring_types = [rings_list[i] for i in atom_type]
    for i in range(x.shape[0]):
        ax.text(x[i, 0], x[i, 1], ring_types[i], fontsize=20, ha="center", va="center")

    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[0]):
            if adj[i, j] == 1:
                ax.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], c="black")

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    return ax


def plot_rdkit(
    x,
    ring_type,
    filename=None,
    showPlot=False,
    title="",
    tol=0.1,
    dataset="cata",
    addInChi=True,
):
    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 9))
    atoms_positions, atoms_types, bonds = gor2goa(x, ring_type, dataset, tol)
    valid, val_ration = rdkit_valid([atoms_types], [bonds], dataset)
    if len(valid) == 0:
        return
    if addInChi:
        title = title + "\n" + valid[0]
    mol = Chem.MolFromInchi(valid[0])
    img = Chem.Draw.MolToImage(mol, size=(3600, 3600))
    Chem.Draw.MolToFile(mol, filename + ".png")
    plt.imshow(img)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])

    # save figure
    if filename:
        fig.savefig(filename, bbox_inches="tight", pad_inches=0.0)

    # show figure
    if showPlot:
        plt.show()
    plt.close()


def plot_graph_of_rings(
    x,
    atom_type,
    filename=None,
    showPlot=False,
    title="",
    tol=0.1,
    axis_lim=10,
    dataset="cata",
    orientation=False,
    adj=None,
):
    # set parameters
    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 9))
    plot_grap_of_rings_inner(
        ax,
        x,
        atom_type,
        title,
        tol=tol,
        axis_lim=axis_lim,
        dataset=dataset,
        orientation=orientation,
        adj=adj,
    )

    # save figure
    if filename:
        fig.savefig(filename, bbox_inches="tight", pad_inches=0.0)

    # show figure
    if showPlot:
        plt.show()
    plt.close()


def plot_rings_orientation(
    x,
    node_mask,
    atom_type,
    adj,
    filename=None,
    showPlot=False,
    title="",
    tol=0.1,
    axis_lim=10,
):
    x = x[node_mask.bool()]
    atom_type = atom_type[node_mask.bool()]
    adj = adj[node_mask.bool()][:, node_mask.bool()]

    # set parameters
    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 9))
    plot_grap_of_rings_inner(
        ax,
        x,
        atom_type,
        title,
        tol=tol,
        axis_lim=axis_lim,
        dataset="hetro",
        orientation=True,
        adj=adj,
    )

    # save figure
    if filename:
        fig.savefig(filename, bbox_inches="tight", pad_inches=0.0)

    # show figure
    if showPlot:
        plt.show()
    plt.close()


def plot_graph_of_rings_list(
    molecule_list, filename=None, showPlot=False, title="", tol=0.1, axis_lim=10
):
    n = len(molecule_list)
    # set parameters
    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(1, n, figsize=(2.5 * n, 3))
    for i, (x, one_hot) in enumerate(molecule_list):
        plot_grap_of_rings_inner(ax[i], x, one_hot, tol=tol, axis_lim=axis_lim)

    fig.suptitle(title)
    # save figure
    if filename:
        fig.savefig(filename, bbox_inches="tight", pad_inches=0.0)

    # show figure
    if showPlot:
        plt.show()
    plt.close()


def plot_graph_of_atoms(
    x, one_hot, adj, filename=None, showPlot=False, title="", tol=0.1, axis_lim=10
):
    # set parameters
    plt.rcParams.update({"font.size": 22})
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 9))
    x = x.cpu().numpy()
    x = align_to_xy_plane(x)

    ax.scatter(x[:, 0], x[:, 1], c="blue")
    atom_types = [ATOMS_LIST["hetro"][i] for i in one_hot.argmax(1)]
    for i in range(x.shape[0]):
        ax.text(x[i, 0], x[i, 1], atom_types[i], fontsize=20, ha="center", va="center")

    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[0]):
            if adj[i, j] == 1:
                ax.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], c="black")

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)

    # save figure
    if filename:
        fig.savefig(filename, bbox_inches="tight", pad_inches=0.0)

    # show figure
    if showPlot:
        plt.show()
    plt.close()


def plot_graph_of_rings_3d(
    x,
    atom_type,
    filename=None,
    showPlot=False,
    title="",
    tol=0.1,
    axis_lim=6,
    dataset="cata",
    orientation=False,
    colors=False,
):
    rings_list = RINGS_LIST["hetro"]
    if orientation:
        n = x.shape[0] // 2
        _, adj = positions2adj(
            x[None, :n, :], atom_type[None, :n], tol=tol, dataset=dataset
        )
        adj = adj[0]
        adj = torch.cat(
            [
                torch.cat([adj, torch.eye(n)], dim=1),
                torch.cat([torch.eye(n), torch.zeros(n, n)], dim=1),
            ],
            dim=0,
        )
    else:
        _, adj = positions2adj(
            x[None, :, :], atom_type[None, :], tol=tol, dataset=dataset
        )
        adj = adj[0]

    x = x.cpu().numpy()

    # set parameters
    plt.rcParams.update({"font.size": 22})

    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(projection="3d")

    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            if adj[i, j] == 1:
                plt.plot(
                    [x[i, 0], x[j, 0]],
                    [x[i, 1], x[j, 1]],
                    [x[i, 2], x[j, 2]],
                    c="black",
                )

    ring_types = [rings_list[i] for i in atom_type]
    if colors:
        palette = plt.get_cmap("gist_rainbow")
        palette = plt.cm.colors.ListedColormap(
            [palette(x) for x in np.linspace(0, 1, 12)]
        ).colors
        # palette = [plt.cm.Paired(i) for i in range(12)]
        c = [palette[i] for i in atom_type]
        ax.scatter(
            xs=x[::-1, 0], ys=x[::-1, 1], zs=x[::-1, 2], c=c[::-1], s=400, alpha=0.8
        )
    else:
        ax.scatter(xs=x[:, 0], ys=x[:, 1], zs=x[:, 2], c="blue", s=100)
        for i in range(x.shape[0]):
            ax.text(
                x[i, 0],
                x[i, 1],
                x[i, 2],
                ring_types[i],
                fontsize=20,
                ha="center",
                va="center",
            )

    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            if adj[i, j] == 1:
                plt.plot(
                    [x[i, 0], x[j, 0]],
                    [x[i, 1], x[j, 1]],
                    [x[i, 2], x[j, 2]],
                    c="black",
                )

    plt.title(title)
    ax.set_axis_off()
    if axis_lim:
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)

    # save figure
    if filename:
        fig.savefig(filename, bbox_inches="tight", pad_inches=0.0)

    # show figure
    if showPlot:
        plt.show()
    plt.close()


def plot_chain(
    x,
    atom_type,
    dirname,
    filename,
    title="",
    tol=0.1,
    axis_lim=6.0,
    dataset="cata",
    orientation=False,
    gif=True,
    colors=False,
):
    save_paths = []
    try:
        os.mkdir(dirname)
    except:
        pass
    for i in range(x.shape[0]):
        save_paths.append(f"{dirname}/chain{i}.pdf")
        plot_graph_of_rings_3d(
            x[i],
            atom_type[i],
            filename=save_paths[-1],
            tol=tol,
            axis_lim=axis_lim,
            dataset=dataset,
            orientation=orientation,
            title=i,
            colors=True,
        )

    if gif:
        # create gif
        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = f"{dirname}/{filename}.gif"
        print(f"Creating gif with {len(imgs)} images")
        # Add the last frame 10 times so that the final result remains temporally.
        # imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True)

        # delete png files
        for file in save_paths:
            os.remove(file)
