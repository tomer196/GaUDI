import os
import random
import sys
from pathlib import Path
from time import time
from typing import Tuple

import networkx as nx
import numpy as np
import torch
import pandas as pd
from torch import zeros, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
from tqdm import tqdm

from data.mol import Mol, load_xyz, from_rdkit
from data.ring import RINGS_DICT
from utils.args_edm import Args_EDM
from utils.ring_graph import get_rings, get_rings_adj
from utils.molgraph import get_connectivity_matrix, get_edges

DTYPE = torch.float32
INT_DTYPE = torch.int8
# ATOMS_LIST = __ATOM_LIST__[:8]
ATOMS_LIST = {
    "cata": ["H", "C"],
    "peri": ["H", "C"],
    "hetro": ["H", "C", "B", "N", "O", "S"],
}
RINGS_LIST = {
    "cata": ["Bn"],
    "peri": ["Bn"],
    "hetro": list(RINGS_DICT.keys()) + ["."],
}


class RandomRotation(object):
    def __call__(self, x):
        M = torch.randn(3, 3)
        Q, __ = torch.linalg.qr(M)
        return x @ Q


class AromaticDataset(Dataset):
    def __init__(self, args, task: str = "train"):
        """
        Args:
            args: All the arguments.
            task: Select the dataset to load from (train/val/test).
        """
        self.csv_file, self.xyz_root = get_paths(args)

        self.task = task
        self.rings_graph = args.rings_graph
        self.normalize = args.normalize
        self.max_nodes = args.max_nodes
        self.return_adj = False
        self.dataset = args.dataset
        self.target_features = getattr(args, "target_features", None)
        self.target_features = (
            self.target_features.split(",") if self.target_features else []
        )
        self.orientation = False if self.dataset == "cata" else True
        self._edge_mask_orientation = None
        self.atoms_list = ATOMS_LIST[self.dataset]
        self.knots_list = RINGS_LIST[self.dataset]

        self.df = getattr(args, f"df_{task}").reset_index()
        self.df = self.df[self.df.n_rings <= args.max_nodes].reset_index()
        if args.normalize:
            train_df = args.df_train
            try:
                target_data = train_df[self.target_features].values
            except:
                self.target_features = [
                    t.replace(" ", "") for t in self.target_features
                ]
                target_data = train_df[self.target_features].values
            self.mean = torch.tensor(target_data.mean(0), dtype=DTYPE)
            self.std = torch.tensor(target_data.std(0), dtype=DTYPE)
        else:
            self.std = torch.ones(1, dtype=DTYPE)
            self.mean = torch.zeros(1, dtype=DTYPE)

        self.examples = np.arange(self.df.shape[0])
        if args.sample_rate < 1:
            random.shuffle(self.examples)
            num_files = round(len(self.examples) * args.sample_rate)
            self.examples = self.examples[:num_files]

        x, node_mask, edge_mask, node_features, y = self.__getitem__(0)[:5]
        self.num_node_features = node_features.shape[1]
        self.num_targets = y.shape[0]

    def get_edge_mask_orientation(self):
        if self._edge_mask_orientation is None:
            self._edge_mask_orientation = torch.zeros(
                2 * self.max_nodes, 2 * self.max_nodes, dtype=torch.bool
            )
            for i in range(self.max_nodes):
                self._edge_mask_orientation[i, self.max_nodes + i] = True
                self._edge_mask_orientation[self.max_nodes + i, i] = True
        return self._edge_mask_orientation.clone()

    def __len__(self):
        return len(self.examples)

    def rescale_loss(self, x):
        # Convert from normalized to the original representation
        if self.normalize:
            x = x * self.std.to(x.device).mean()
        return x

    def get_mol(self, df_row, skip_hydrogen=False) -> Tuple[Mol, list, Tensor, str]:
        name = df_row["molecule"]
        file_path = self.xyz_root + "/" + name
        if os.path.exists(file_path + ".xyz"):
            mol = load_xyz(file_path + ".xyz")
            atom_connectivity = get_connectivity_matrix(
                mol.atoms, skip_hydrogen=skip_hydrogen
            )  # build connectivity matrix
            # edges = bonds
        elif os.path.exists(file_path + ".pkl"):
            mol, atom_connectivity = from_rdkit(file_path + ".pkl")
        else:
            raise NotImplementedError(file_path)
        edges = get_edges(atom_connectivity)
        return mol, edges, atom_connectivity, name

    def get_rings(self, df_row):
        name = df_row["molecule"]
        preprocessed_path = self.xyz_root + "_rings_preprocessed/" + name + ".xyz"
        if Path(preprocessed_path).is_file():
            if self.orientation:
                x, adj, node_features, orientation = torch.load(preprocessed_path)
            else:
                x, adj, node_features = torch.load(preprocessed_path)
                orientation = False
        else:
            mol, edges, atom_connectivity, _ = self.get_mol(df_row, skip_hydrogen=True)
            # get_figure(mol, edges, showPlot=True, filename='4.png')
            mol_graph = nx.Graph(edges)
            knots = get_rings(mol.atoms, mol_graph)
            adj = get_rings_adj(knots)
            x = torch.tensor([k.get_coord() for k in knots], dtype=DTYPE)
            knot_type = torch.tensor(
                [self.knots_list.index(k.cycle_type) for k in knots]
            ).unsqueeze(1)
            node_features = (
                one_hot(knot_type, num_classes=len(self.knots_list)).squeeze(1).float()
            )
            orientation = [k.orientation for k in knots]
            torch.save([x, adj, node_features, orientation], preprocessed_path)
        return x, adj, node_features, orientation

    def get_atoms(self, df_row):
        name = df_row["molecule"]
        preprocessed_path = self.xyz_root + "_atoms_preprocessed/" + name + ".xyz"
        if Path(preprocessed_path).is_file():
            x, adj, node_features = torch.load(preprocessed_path)
        else:
            mol, edges, atom_connectivity, _ = self.get_mol(df_row)
            # get_figure(mol, edges, showPlot=True)
            x = torch.tensor([a.get_coord() for a in mol.atoms], dtype=DTYPE)
            atom_element = torch.tensor(
                [self.atoms_list.index(atom.element) for atom in mol.atoms]
            ).unsqueeze(1)
            node_features = (
                one_hot(atom_element, num_classes=len(self.atoms_list))
                .squeeze(1)
                .float()
            )
            adj = atom_connectivity
            torch.save([x, adj, node_features], preprocessed_path)
        return x, adj, node_features

    def get_all(self, df_row):
        # extract targets
        y = torch.tensor(
            df_row[self.target_features].values.astype(np.float32), dtype=DTYPE
        )
        if self.normalize:
            y = (y - self.mean) / self.std

        # creation of nodes, edges and there features
        x, adj, node_features, orientation = self.get_rings(df_row)

        if self.orientation:
            # adjust to max nodes shape
            n_nodes = x.shape[0]
            x_r = torch.tensor([random.sample(o, 1)[0] for o in orientation])
            x_full = zeros(self.max_nodes * 2, 3)
            x_full[:n_nodes] = x
            x_full[self.max_nodes : self.max_nodes + n_nodes] = x_r

            node_mask = zeros(self.max_nodes * 2)
            node_mask[:n_nodes] = 1
            node_mask[self.max_nodes : self.max_nodes + n_nodes] = 1

            node_features_full = zeros(self.max_nodes * 2, node_features.shape[1])
            node_features_full[:n_nodes, :] = node_features
            # mark the orientation nodes as additional ring type
            node_features_full[self.max_nodes : self.max_nodes + n_nodes, -1] = 1

            edge_mask_tmp = node_mask[: self.max_nodes].unsqueeze(0) * node_mask[
                : self.max_nodes
            ].unsqueeze(1)
            # mask diagonal
            diag_mask = ~torch.eye(self.max_nodes, dtype=torch.bool)
            edge_mask_tmp *= diag_mask
            edge_mask = self.get_edge_mask_orientation()
            edge_mask[: self.max_nodes, : self.max_nodes] = edge_mask_tmp

            if self.return_adj:
                adj_full = self.get_edge_mask_orientation()
                adj_full[:n_nodes, :n_nodes] = adj
        else:
            # adjust to max nodes shape
            n_nodes = x.shape[0]
            x_full = zeros(self.max_nodes, 3)

            node_mask = zeros(self.max_nodes)
            x_full[:n_nodes] = x
            node_mask[:n_nodes] = 1

            node_features_full = zeros(self.max_nodes, node_features.shape[1])
            node_features_full[:n_nodes, :] = node_features
            # node_features_full = zeros(self.max_nodes, 0)

            # edge_mask = zeros(self.max_nodes, self.max_nodes)
            # edge_mask[:n_nodes, :n_nodes] = adj
            # edge_mask = edge_mask.view(-1, 1)

            edge_mask = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
            # mask diagonal
            diag_mask = ~torch.eye(self.max_nodes, dtype=torch.bool)
            edge_mask *= diag_mask
            # edge_mask = edge_mask.view(-1, 1)

            if self.return_adj:
                adj_full = zeros(self.max_nodes, self.max_nodes)
                adj_full[:n_nodes, :n_nodes] = adj

        if self.return_adj:
            return x_full, node_mask, edge_mask, node_features_full, adj_full, y
        else:
            return x_full, node_mask, edge_mask, node_features_full, y

    def __getitem__(self, idx):
        index = self.examples[idx]
        df_row = self.df.loc[index]
        return self.get_all(df_row)


def get_paths(args):
    if not hasattr(args, "dataset"):
        csv_path = args.csv_file
        xyz_path = args.xyz_root
    elif args.dataset == "cata":
        csv_path = "/home/tomerweiss/PBHs-design/data/COMPAS-1x.csv"
        xyz_path = "/home/tomerweiss/PBHs-design/data/peri-cata-89893-xyz"
    elif args.dataset == "peri":
        csv_path = "/home/tomerweiss/PBHs-design/data/peri-xtb-data-55821.csv"
        xyz_path = "/home/tomerweiss/PBHs-design/data/peri-cata-89893-xyz"
    elif args.dataset == "hetro":
        csv_path = "/home/tomerweiss/PBHs-design/data/db-474K-OPV-phase-2-filtered.csv"
        xyz_path = "/home/tomerweiss/PBHs-design/data/db-474K-xyz"
    elif args.dataset == "hetro-dft":
        csv_path = "/home/tomerweiss/PBHs-design/data/db-15067-dft.csv"
        xyz_path = ""
    else:
        raise NotImplementedError
    return csv_path, xyz_path


def get_splits(args, random_seed=42, val_frac=0.1, test_frac=0.1):
    np.random.seed(seed=random_seed)
    csv_path, _ = get_paths(args)
    if hasattr(args, "dataset") and args.dataset == "hetro":
        targets = (
            args.target_features.split(",")
            if getattr(args, "target_features", None) is not None
            else []
        )
        df = pd.read_csv(csv_path, usecols=["name", "nRings", "inchi"] + targets)
        df.rename(columns={"nRings": "n_rings", "name": "molecule"}, inplace=True)
        args.max_nodes = min(args.max_nodes, 10)
    else:
        df = pd.read_csv(csv_path)

    df_all = df.copy()
    df_test = df.sample(frac=test_frac, random_state=random_seed)
    df = df.drop(df_test.index)
    df_val = df.sample(frac=val_frac, random_state=random_seed)
    df_train = df.drop(df_val.index)
    return df_train, df_val, df_test, df_all


def create_data_loaders(args):
    args.df_train, args.df_val, args.df_test, args.df_all = get_splits(args)

    train_dataset = AromaticDataset(
        args=args,
        task="train",
    )
    val_dataset = AromaticDataset(
        args=args,
        task="val",
    )
    test_dataset = AromaticDataset(
        args=args,
        task="test",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    args = Args_EDM().parse_args()
    # args.dataset = "cata"
    # args.target_features = "GAP_eV"
    train_loader, val_loader, test_loader = create_data_loaders(args)
    # import matplotlib.pyplot as plt
    # for i in range(5):
    #     mol, edges = dataset.get_mol(i)
    #     fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    #     ax.axis('off')
    #
    #     # plot molecule
    #     moldraw(ax, mol, edges)
    #     ax.set_title()
    #     fig.show()
    # s=time()
    # print(train_loader.dataset[0])
    # print(time()-s)
    #
    times = []
    s = time()
    with tqdm(test_loader) as tepoch:
        for i, data in enumerate(tepoch):
            times.append(time() - s)
            # print(data[0].shape)
            s = time()
            # if i==2000:
            #     print(np.mean(times), np.std(times))
            #     sys.exit()
    print(np.mean(times), np.std(times))
