from torch import Tensor
from tqdm import tqdm

import torch
import networkx as nx
import numpy as np

from data.aromatic_dataloader import create_data_loaders, RINGS_LIST
from edm.qm9.analyze import Histogram_discrete, Histogram_cont, kl_divergence_sym
from utils.args_edm import Args_EDM
from utils.utils import (
    positions2adj,
    coord2distances,
    ring_distances,
    angels3_dict,
    angels4_dict,
)
import pandas as pd


def check_angels3(angels3: Tensor, tol=0.1, dataset="cata") -> bool:
    a3_dict = angels3_dict[dataset]
    if len(angels3) == 0:
        return True
    symbols = [a[0] for a in angels3]
    for symbol in set(symbols):
        a3_symbol = torch.stack([a[1] for a in angels3 if a[0] == symbol])
        conds = [
            torch.logical_and(
                q_low * (1 - tol) <= a3_symbol, a3_symbol <= q_high * (1 + tol)
            )
            for q_low, q_high in a3_dict[symbol].values()
        ]
        if not torch.stack(conds).any(dim=0).all():
            return False
    return True


def check_angels4(angels4: Tensor, tol=0.1, dataset="cata") -> bool:
    if len(angels4) == 0 or dataset == "hetro":
        return True
    a4_dict = angels4_dict[dataset]
    angels4 = torch.stack([a for s, a in angels4])
    cond = torch.logical_or(
        a4_dict["180"] * (1 - tol) <= angels4, angels4 <= a4_dict["0"] * (1 + tol)
    )
    return cond.all()


def check_stability(
    positions, ring_type, tol=0.1, dataset="cata", orientation=False
) -> dict:
    results = {
        "orientation_nodes": True,
        "dist_stable": False,
        "connected": False,
        "angels3": False,
        "angels4": False,
    }
    if isinstance(positions, np.ndarray):
        positions = Tensor(positions)
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    if len(ring_type.shape) == 2:
        ring_type = ring_type.argmax(1)
    if orientation:
        n_rings = torch.div(positions.shape[0], 2, rounding_mode="trunc")
        positions = positions[:n_rings]
        # check orientation nodes
        orientation_ring_type = len(RINGS_LIST["hetro"]) - 1
        if (
            set(ring_type[n_rings:].numpy()) != set([orientation_ring_type])
            or orientation_ring_type in ring_type[:n_rings]
        ):
            results["orientation_nodes"] = False
            return results
        ring_type = ring_type[:n_rings]
    n_rings = positions.shape[0]
    dist, adj = positions2adj(
        positions[None, :, :], ring_type[None, :], tol, dataset=dataset
    )
    dist = dist[0]
    adj = adj[0]
    min_dist = min([r[0] for r in ring_distances[dataset].values()])
    if ((dist < min_dist * (1 - tol)) * (1 - torch.eye(n_rings))).bool().any():
        return results
    else:
        results["dist_stable"] = True

    g = nx.from_numpy_matrix(adj.numpy())
    if not nx.is_connected(g):
        return results
    else:
        results["connected"] = True

    angels3, angels4 = get_angels(
        positions[None, :, :], ring_type[None, :], adj[None, :, :], dataset=dataset
    )
    results["angels3"] = check_angels3(angels3, tol, dataset=dataset)
    results["angels4"] = check_angels4(angels4, tol, dataset=dataset)
    # plot_graph_of_rings(positions, ring_type, filename=f'tests/test.png',
    #                     tol=0.1)
    return results


def main_check_stability(args, tol=0.1):
    args.sample_rate = 0.1
    train_loader, _, _ = create_data_loaders(args)
    train_dataloader, val_dataloader, test_dataloader = create_data_loaders(args)

    def test_validity_for(dataloader):
        molecule_list = []
        for i, (x, node_mask, edge_mask, node_features, y) in enumerate(dataloader):
            for j in range(x.size(0)):
                positions = x[j].view(-1, 3)
                one_hot = node_features[j].type(torch.float32)
                mask = node_mask[j].flatten().bool()
                positions, one_hot = positions[mask], one_hot[mask]
                atom_type = torch.argmax(one_hot, dim=1).numpy()
                molecule_list.append((positions, atom_type))
        validity_dict, molecule_stable_list = analyze_stability_for_molecules(
            molecule_list, tol=tol, dataset=args.dataset, orientation=args.orientation
        )
        del validity_dict["molecule_stable_bool"]
        print(validity_dict)

    print("For train")
    test_validity_for(train_dataloader)
    print("For val")
    test_validity_for(val_dataloader)
    print("For test")
    test_validity_for(test_dataloader)


def analyze_stability_for_molecules(
    molecule_list, tol=0.1, dataset="cata", orientation=False
):
    n_samples = len(molecule_list)
    molecule_stable_list = []
    molecule_stable_bool = []

    n_molecule_stable = (
        n_dist_stable
    ) = n_connected = n_angel3 = n_angel4 = n_orientation = 0

    # with tqdm(molecule_list, unit="mol") as tq:
    for i, (x, atom_type) in enumerate(molecule_list):
        validity_results = check_stability(
            x, atom_type, tol=tol, dataset=dataset, orientation=orientation
        )

        molecule_stable = all(validity_results.values())
        n_molecule_stable += int(molecule_stable)
        n_dist_stable += int(validity_results["dist_stable"])
        n_connected += int(validity_results["connected"])
        n_angel3 += int(validity_results["angels3"])
        n_angel4 += int(validity_results["angels4"])
        n_orientation += int(validity_results["orientation_nodes"])

        molecule_stable_bool.append(molecule_stable)
        if molecule_stable:
            molecule_stable_list.append((x, atom_type))

    # Validity
    validity_dict = {
        "mol_stable": n_molecule_stable / float(n_samples),
        "orientation_nodes": n_orientation / float(n_samples),
        "dist_stable": n_dist_stable / float(n_samples),
        "connected": n_connected / float(n_samples),
        "angels3": n_angel3 / float(n_samples),
        "angels4": n_angel4 / float(n_samples),
        "molecule_stable_bool": molecule_stable_bool,
    }

    # print('Validity:', validity_dict)

    return validity_dict, molecule_stable_list


def angel3(p):
    v1 = p[0] - p[1]
    v2 = p[2] - p[1]
    dot_product = torch.dot(v1, v2)
    norm_product = torch.norm(v1) * torch.norm(v2)
    a = torch.rad2deg(torch.acos(dot_product / norm_product))
    return a if a >= 0 else a + 360


def angel4(p):
    """Praxeolitic formula for dihedral angle"""
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= torch.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - torch.dot(b0, b1) * b1
    w = b2 - torch.dot(b2, b1) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = torch.dot(v, w)
    y = torch.dot(torch.cross(b1, v), w)
    return torch.rad2deg(torch.atan2(y, x)).abs()


def find_triplets_quads(adj: Tensor, x: Tensor, ring_types: Tensor, dataset="cata"):
    rings_list = RINGS_LIST[dataset]
    if len(ring_types.shape) == 2:
        ring_types = ring_types.argmax(1)
    rings = [rings_list[i] for i in ring_types]
    g = nx.from_numpy_matrix(adj.numpy())
    triplets = []
    for n1, n2 in nx.bfs_edges(g, 0):
        for n3 in g.neighbors(n1):
            if n3 != n2:
                triplets.append((n2, n1, n3))
        for n3 in g.neighbors(n2):
            if n3 != n1:
                triplets.append((n1, n2, n3))
    triplets = [(n1, n2, n3) if n1 < n3 else (n3, n2, n1) for n1, n2, n3 in triplets]
    triplets = list(set(triplets))
    # save all the angels with the center ring type
    # if any([angel3(x[nodes, :])<60 for nodes in triplets]):
    #     print([angel3(x[nodes, :]) for nodes in triplets])
    angels3 = [(rings[nodes[1]], angel3(x[nodes, :])) for nodes in triplets]

    angular_triplets = [t for t in triplets if not 170 < angel3(x[t, :]) < 190]
    # quads
    quads = []
    for n1, n2, n3 in angular_triplets:
        for n4 in g.neighbors(n1):
            if n4 not in [n2, n3]:
                # check the new angle is not linear
                if not 175 < angel3(x[[n4, n1, n2]]) < 185:
                    quads.append((n4, n1, n2, n3))
        for n4 in g.neighbors(n3):
            if n4 not in [n1, n2]:
                # check the new angle is not linear
                if not 175 < angel3(x[[n2, n3, n4]]) < 185:
                    quads.append((n1, n2, n3, n4))
    quads = [
        (n1, n2, n3, n4) if n1 < n4 else (n4, n3, n2, n1) for n1, n2, n3, n4 in quads
    ]
    quads = list(set(quads))
    angels4 = [
        ([rings[nodes[i]] for i in range(4)], angel4(x[nodes, :])) for nodes in quads
    ]

    # if any([80<a[1]<100 for a in angels4]):
    #     print([a[1] for a in angels4])

    return angels3, angels4


def get_angels(xs: Tensor, ring_types, adjs, node_masks=None, dataset="cata"):
    """Extract list of angels from batch of nodes"""
    # _, adjs = positions2adj(xs, ring_types, dataset)
    angels3 = []
    angels4 = []
    for i in range(xs.shape[0]):
        adj = adjs[i]
        x = xs[i]
        ring_type = ring_types[i]
        if node_masks is not None:
            node_mask = node_masks[i].bool()
            adj = adj[node_mask][:, node_mask]
            x = x[node_mask]
            ring_type = ring_type[node_mask]
        a3, a4 = find_triplets_quads(adj, x, ring_type, dataset)
        angels3 += a3
        angels4 += a4

    return angels3, angels4


def main_analyze_rings(args):
    # args.batch_size = 1
    args.sample_rate = 0.1
    # args.num_workers = 0
    train_loader, _, _ = create_data_loaders(args)
    train_loader.dataset.return_adj = True

    hist_rings = Histogram_discrete("Histogram # rings")
    hist_ring_type = Histogram_discrete("Histogram of ring types")
    hist_dist = Histogram_cont(
        name="Histogram relative distances", ignore_zeros=True, range=[0.0, 20]
    )
    neighbor_hist_dist = Histogram_cont(
        name="Histogram relative distances of neighbors",
        ignore_zeros=True,
        range=[0.0, 20],
    )
    no_neighbor_hist_dist = Histogram_cont(
        name="Histogram relative distances of non neighbors",
        ignore_zeros=True,
        range=[0.0, 20],
    )
    neighbor_hist_angle3 = Histogram_cont(
        name="Histogram angle between neighbors", ignore_zeros=True, range=[0.0, 360]
    )
    neighbor_hist_angle4 = Histogram_cont(
        name="Histogram angle between neighbors", ignore_zeros=True, range=[0.0, 360]
    )
    n_dist = []
    angels3_all = []
    angels4_all = []

    with tqdm(train_loader, unit="batch") as tepoch:
        for i, data in enumerate(tepoch):
            # print(i, len(train_loader))
            x, node_mask, edge_mask, node_features, adj, y = data

            # Histogram num_nodes
            num_nodes = torch.sum(node_mask, dim=1)
            num_nodes = list(num_nodes.numpy())
            hist_rings.add(num_nodes)

            # Histogram edge distances
            x = x * node_mask.unsqueeze(2)
            dist = coord2distances(x)
            hist_dist.add(list(dist.flatten().numpy()))

            # Histogram edge distances
            neighbor_dist = (dist * adj).flatten()
            neighbor_hist_dist.add(list(neighbor_dist.numpy()))
            n_dist.append(neighbor_dist[neighbor_dist != 0])
            no_neighbor_dist = (dist * (1 - adj)).flatten()
            no_neighbor_hist_dist.add(list(no_neighbor_dist.numpy()))

            # Histogram of atom types
            one_hot = node_features.float()
            atom = torch.argmax(one_hot, 2)
            atom = atom.flatten()
            mask = node_mask.flatten()
            masked_atoms = list(atom[mask.long()].numpy())
            hist_ring_type.add(masked_atoms)

            # Histogram of angels
            angels3, angels4 = get_angels(x, one_hot, adj, node_mask)
            angels3 = [t[1] for t in angels3]
            neighbor_hist_angle3.add(angels3)
            neighbor_hist_angle4.add(angels4)
            angels3_all.append(angels3)
            angels4_all.append(angels4)

    hist_dist.plot(f"data/hist_dist_{args.dataset}.png")
    hist_dist.plot_both(hist_dist.bins[::-1])
    print(
        "KL divergence A %.4f" % kl_divergence_sym(hist_dist.bins, hist_dist.bins[::-1])
    )
    print("KL divergence B %.4f" % kl_divergence_sym(hist_dist.bins, hist_dist.bins))
    print({k.item(): v for k, v in zip(hist_dist.bins_border[:-1], hist_dist.bins)})
    print(hist_dist.bins)
    hist_rings.plot(f"data/hist_rings_{args.dataset}.png")
    print(hist_rings.bins)
    hist_ring_type.plot(
        f"data/hist_ring_type_{args.dataset}.png", xticks=RINGS_LIST[args.dataset]
    )
    print(hist_ring_type.bins)

    neighbor_hist_dist.plot(f"data/neighbor_hist_dist_{args.dataset}.png")
    no_neighbor_hist_dist.plot(f"data/no_neighbor_hist_dist_{args.dataset}.png")
    neighbor_hist_dist.plot_both(
        no_neighbor_hist_dist.bins, f"data/neighbors_{args.dataset}.png"
    )
    print(f"Neighbor dist mean: {torch.cat(n_dist).mean()}")
    print(f"Neighbor dist max: {torch.cat(n_dist).max()}")
    print(f"Neighbor dist min: {torch.cat(n_dist).min()}")

    neighbor_hist_angle3.plot(f"data/neighbor_hist_angle_{args.dataset}.png")
    neighbor_hist_angle4.plot(f"data/neighbor_hist_angle_{args.dataset}.png")
    torch.save(torch.cat(angels3_all), f"analyze/angels3_{args.dataset}.pt")
    torch.save(torch.cat(angels4_all), f"analyze/angels4_{args.dataset}.pt")
    torch.save(torch.cat(n_dist), f"analyze/dists_{args.dataset}.pt")


def main_analyze_rings_hetro(args):
    # args.batch_size = 128
    args.sample_rate = 0.1
    # args.num_workers = 0
    train_loader, _, _ = create_data_loaders(args)
    train_loader.dataset.return_adj = True

    hist_rings = Histogram_discrete("Histogram # rings")
    dists = []
    angels3_all = []
    angels4_all = []

    with tqdm(train_loader, unit="batch") as tepoch:
        for i, data in enumerate(tepoch):
            # print(i, len(train_loader))
            x, node_mask, _, node_features, adj, y = data
            if args.orientation:
                n = x.shape[1] // 2
                x = x[:, :n, :]
                node_mask = node_mask[:, :n]
                node_features = node_features[:, :n, :]
                adj = adj[:, :n, :n]
            # Histogram num_nodes
            num_nodes = torch.sum(node_mask, dim=1)
            num_nodes = list(num_nodes.numpy())
            hist_rings.add(num_nodes)

            # Histogram edge distances
            x = x * node_mask.unsqueeze(2)
            dist = coord2distances(x)

            # Histogram edge distances
            neighbor_dist = (dist * adj).triu()
            bonds_index = torch.nonzero(neighbor_dist)
            atom = node_features.float().argmax(2)
            bonds_symbols1 = [
                atom[bonds_index[i, 0], bonds_index[i, 1]]
                for i in range(bonds_index.shape[0])
            ]
            bonds_symbols2 = [
                atom[bonds_index[i, 0], bonds_index[i, 2]]
                for i in range(bonds_index.shape[0])
            ]
            neighbor_dist = neighbor_dist.flatten()
            neighbor_dist = neighbor_dist[neighbor_dist != 0]
            dists += list(zip(bonds_symbols1, bonds_symbols2, neighbor_dist))

            # Histogram of angels
            angels3, angels4 = get_angels(
                x, node_features, adj, node_mask, dataset=args.dataset
            )
            angels3_all += angels3
            angels4_all += angels4

    # hist_rings.plot(f'data/hist_rings_{args.dataset}.png')
    # print(hist_rings.bins)

    # torch.save(dists, f'analyze/dists_{args.dataset}.pt')
    df = pd.DataFrame(dists, columns=["from", "to", "dist"]).astype(float)

    knots = RINGS_LIST[args.dataset]
    mapping = {i: k for i, k in enumerate(knots)}
    df["from"] = df["from"].astype(int).map(mapping)
    df["to"] = df["to"].astype(int).map(mapping)
    df.to_pickle(f"analyze/dists_{args.dataset}.pkl")

    a3_symbol = [a[0] for a in angels3_all]
    a3 = [a[1] for a in angels3_all]
    df_angels3 = pd.DataFrame(list(zip(a3_symbol, a3)), columns=["symbol", "angel"])
    df_angels3.to_pickle(f"analyze/angels3_{args.dataset}.pkl")

    a4_symbol = [a[0] for a in angels4_all]
    a4 = [a[1] for a in angels4_all]
    df_angels4 = pd.DataFrame(list(zip(a4_symbol, a4)), columns=["symbols", "angel"])
    df_angels4[["1", "2", "3", "4"]] = pd.DataFrame(
        df_angels4.symbols.tolist(), index=df_angels4.index
    )
    df_angels4.drop(columns=["symbols"], inplace=True)
    df_angels4.to_pickle(f"analyze/angels4_{args.dataset}.pkl")
    # torch.save(angels4_all, f'analyze/angels4_{args.dataset}.pt')


if __name__ == "__main__":
    args = Args_EDM().parse_args()
    args.dataset = "cata"
    args.orientation = False
    # main_analyze_rings(args)
    # main_analyze_rings_hetro(args)
    main_check_stability(args)
