import argparse
import os
import json
import random
import shutil
from datetime import datetime
from time import time, sleep
import warnings

import sys, os

sys.path.append("/home/tomerweiss/PBHs-design")

import matplotlib.pyplot as plt
from torch.nn.functional import l1_loss, mse_loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from edm.egnn_predictor.models import EGNN_predictor
from edm.equivariant_diffusion.utils import (
    remove_mean_with_mask,
    assert_correctly_masked,
    assert_mean_zero_with_mask,
)
from data.aromatic_dataloader import AromaticDataset, get_splits, get_paths, DTYPE
from models_edm import get_model, MyDataParallel
import numpy as np
import pandas as pd
import torch
from torch import optim, Tensor, linspace

from train_cond_predictor import check_mask_correct

warnings.simplefilter(action="ignore", category=FutureWarning)


def compute_loss(model, x, h, node_mask, edge_mask, target, y_orig):
    xh = torch.cat([x, h], dim=2)

    bs, n_nodes, n_dims = x.size()
    assert_correctly_masked(x, node_mask)
    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
    pred = model(xh, node_mask, edge_mask)
    # pred = pred + y_orig.to(pred.device)
    loss = l1_loss(pred, target)
    return loss, (pred - target).abs().detach()


def train_epoch(epoch, predictor, dataloader, optimizer, args, writer):
    print("Withoht residual")
    predictor.train()
    start_time = time()
    loss_list = []
    rl_loss = []
    with tqdm(dataloader, unit="batch", desc=f"Train {epoch}") as tepoch:
        for i, (x, node_mask, edge_mask, node_features, y, y_orig) in enumerate(tepoch):
            x = x.to(args.device)
            y = y.to(args.device)
            node_mask = node_mask.to(args.device).unsqueeze(2)
            edge_mask = edge_mask.to(args.device)
            h = node_features.to(args.device)

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, h], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            loss, _ = compute_loss(predictor, x, h, node_mask, edge_mask, y, y_orig)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            rl_loss.append(dataloader.dataset.rescale_loss(loss).item())

            tepoch.set_postfix(loss=np.mean(loss_list).item())
    print(
        f"[{epoch}|train] loss: {np.mean(loss_list):.4f}+-{np.std(loss_list):.4f}, "
        f"L1 (rescaled): {np.mean(rl_loss):.4f}, "
        f" in {int(time()-start_time)} secs"
    )
    sleep(0.01)
    writer.add_scalar("Train loss", np.mean(loss_list), epoch)
    writer.add_scalar("Train L1 (rescaled)", np.mean(rl_loss), epoch)


def val_epoch(
    tag,
    epoch,
    predictor,
    dataloader,
    args,
    writer,
):
    predictor.eval()
    with torch.no_grad():
        start_time = time()
        loss_list = []
        rl_loss = []
        # with tqdm(dataloader, unit="batch", desc=f"{tag} {epoch}") as tepoch:
        for i, (x, node_mask, edge_mask, node_features, y, y_orig) in enumerate(
            dataloader
        ):
            x = x.to(args.device)
            y = y.to(args.device)
            node_mask = node_mask.to(args.device).unsqueeze(2)
            edge_mask = edge_mask.to(args.device)
            h = node_features.to(args.device)

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, h], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            loss, _ = compute_loss(predictor, x, h, node_mask, edge_mask, y, y_orig)

            loss_list.append(loss.item())
            rl_loss.append(dataloader.dataset.rescale_loss(loss).item())

            # tepoch.set_postfix(loss=np.mean(loss_list).item())
        print(
            f"[{epoch}|{tag}] loss: {np.mean(loss_list):.4f}+-{np.std(loss_list):.4f}, "
            f"L1 (rescaled): {np.mean(rl_loss):.4f}, "
            f" in {int(time() - start_time)} secs"
        )
        sleep(0.01)
        writer.add_scalar(f"{tag} loss", np.mean(loss_list), epoch)
        writer.add_scalar(f"{tag} L1 (rescaled)", np.mean(rl_loss), epoch)

    return np.mean(loss_list)


def get_predictor_model(args, dataset: AromaticDataset):
    predictor = EGNN_predictor(
        in_nf=dataset.num_node_features,
        device=args.device,
        hidden_nf=args.nf,
        out_nf=dataset.num_targets,
        act_fn=torch.nn.SiLU(),
        n_layers=args.n_layers,
        recurrent=True,
        tanh=args.tanh,
        attention=args.attention,
        coords_range=args.coords_range,
    )

    if args.dp:  # and torch.cuda.device_count() > 1:
        print(f"Training using {torch.cuda.device_count()} GPUs")
        predictor = MyDataParallel(predictor)
    if args.restore is not None:
        model_state_dict = torch.load(args.exp_dir + "/model.pt")
        predictor.load_state_dict(model_state_dict)
    return predictor


class DeltaMLDataset(AromaticDataset):
    def __init__(self, args, task: str = "train"):
        target_csv_path, _ = get_paths(argparse.Namespace(dataset=args.target_dataset))
        assert args.target_dataset == "hetro-dft"
        targets = args.target_features.split(",")
        replace_dict = {"name": "molecule"}
        replace_dict.update({t.lower(): t for t in targets})
        df = pd.read_csv(
            target_csv_path, usecols=["name"] + [t.lower() for t in targets]
        )
        df.rename(columns=replace_dict, inplace=True)
        args.max_nodes = min(args.max_nodes, 10)
        self.target_df = df.dropna()
        self.target_features = args.target_features.split(",")
        if args.normalize:
            train_df = self.target_df
            target_data = train_df[self.target_features].values
            self.target_mean = torch.tensor(target_data.mean(0), dtype=DTYPE)
            self.target_std = torch.tensor(target_data.std(0), dtype=DTYPE)
        else:
            self.target_std = torch.ones(1, dtype=DTYPE)
            self.target_mean = torch.ones(1, dtype=DTYPE)

        self.df_shrinked = False
        super().__init__(args, task)

    def lazy_shrink_df(self):
        if not self.df_shrinked:
            self.df = self.df[
                self.df.molecule.isin(self.target_df.molecule)
            ].reset_index(drop=True)
            self.examples = np.arange(self.df.shape[0])
            self.df_shrinked = True

    def __getitem__(self, idx):
        self.lazy_shrink_df()
        index = self.examples[idx]
        df_row = self.df.loc[index]
        x, node_mask, edge_mask, node_features, y = self.get_all(df_row)

        y_features = y[None, :].repeat(node_features.shape[0], 1) * node_mask[:, None]
        node_features = torch.cat([node_features, y_features], dim=1)

        df_target_row = self.target_df[self.target_df.molecule == df_row.molecule]
        y_target = torch.tensor(
            df_target_row[self.target_features].values.astype(np.float32), dtype=DTYPE
        ).flatten()
        if self.normalize:
            y = y * self.std + self.mean  # bring y to the same dist as y_target
            y = (y - self.target_mean) / self.target_std
            y_target = (y_target - self.target_mean) / self.target_std

        return x, node_mask, edge_mask, node_features, y_target, y


def create_data_loaders(args):
    args.df_train, args.df_val, args.df_test, args.df_all = get_splits(args)

    train_dataset = DeltaMLDataset(
        args=args,
        task="train",
    )
    val_dataset = DeltaMLDataset(
        args=args,
        task="val",
    )
    test_dataset = DeltaMLDataset(
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


def plot_correlations(dataset):
    df = dataset.df.sort_values("molecule")
    x = df[dataset.target_features].values
    target_df = dataset.target_df[dataset.target_df.molecule.isin(dataset.df.molecule)]
    target_df = target_df.sort_values("molecule")
    y = target_df[dataset.target_features].values
    for i, target in enumerate(dataset.target_features):
        plt.scatter(x[:, i], y[:, i])
        plt.title(target)
        plt.xlabel("xTB")
        plt.ylabel("DFT")
        plt.show()


def main(pred_args):
    # Prepare data
    train_loader, val_loader, test_loader = create_data_loaders(pred_args)
    # plot_correlations(train_loader.dataset)
    predictor = get_predictor_model(pred_args, train_loader.dataset)

    optimizer = optim.AdamW(
        predictor.parameters(), lr=pred_args.lr, amsgrad=True, weight_decay=1e-12
    )

    # Save path
    writer = SummaryWriter(log_dir=pred_args.exp_dir)

    # Run training
    print("Begin training")
    best_val_loss = 1e9
    best_epoch = 0
    for epoch in range(pred_args.num_epochs):
        train_epoch(epoch, predictor, train_loader, optimizer, pred_args, writer)
        val_loss = val_epoch("val", epoch, predictor, val_loader, pred_args, writer)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(predictor.state_dict(), pred_args.exp_dir + "/model.pt")
        if epoch % 50 == 0 and epoch != 0:
            shutil.copy(
                pred_args.exp_dir + "/model.pt", pred_args.exp_dir + f"/model{epoch}.pt"
            )

    print(f"{best_epoch=}, {best_val_loss=:.4f}")
    predictor.load_state_dict(torch.load(pred_args.exp_dir + "/model.pt"))
    print("Test:")
    val_epoch("test", epoch, predictor, test_loader, pred_args, writer)
    writer.close()


class DeltaMLArgs(argparse.ArgumentParser):
    def __init__(
        self,
    ):
        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # data param
        self.add_argument("--dataset", default="hetro", type=str)
        self.add_argument("--target_dataset", default="hetro-dft", type=str)
        self.add_argument("--rings_graph", type=bool, default=True)
        self.add_argument("--max-nodes", default=11, type=str)
        self.add_argument("--orientation", default=True, type=str)
        # task param
        self.add_argument(
            "--target_features",
            # default="HOMO-LUMO gap/eV,HOMO,electron_affinity e/V,ionization e/V,atomisation energy per electron in kcal/mol",
            default="electronic energy,HOMO,LUMO,HOMO-LUMO gap/eV,HOMO-1,LUMO+1",
            # default="LUMO_eV,GAP_eV,Erel_eV,aIP_eV,aEA_eV",
            type=str,
            help="list of the names of the target features in the csv file - can be multiple targets seperated with commas"
            "[HOMO_eV, LUMO_eV, GAP_eV, Dipmom_Debye, Etot_eV, Etot_pos_eV,"
            "Etot_neg_eV, aEA_eV, aIP_eV, Erel_eV]",
        )
        self.add_argument("--sample-rate", type=float, default=1.0)
        self.add_argument("--num-workers", type=int, default=32)

        # training param
        self.add_argument(
            "--name",
            type=str,
            default="hetro2dft",
            # default="hetro_gap_homo_ea_ip_stability_polynomial_2_with_norm",
            # default="cata_lumo_gap_erel_ip_ea_polynomial_2_with_norm",
        )
        self.add_argument("--restore", type=bool, default=None)
        self.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
        self.add_argument("--num_epochs", type=int, default=1000)
        self.add_argument("--normalize", type=bool, default=True)
        self.add_argument("--augmentation", type=bool, default=False)

        self.add_argument("--batch-size", type=int, default=256)

        # Model parameters
        self.add_argument("--dp", type=eval, default=True, help="Data parallelism")
        self.add_argument("--n_layers", type=int, default=12, help="number of layers")
        self.add_argument("--nf", type=int, default=196, help="number of layers")
        self.add_argument("--tanh", type=eval, default=True)
        self.add_argument("--attention", type=eval, default=True)
        self.add_argument("--coords_range", type=float, default=4)

        # Logging
        self.add_argument("--save_dir", type=str, default="summary_delta_ML/")


if __name__ == "__main__":
    pred_args = DeltaMLArgs().parse_args()
    pred_args.exp_dir = f"{pred_args.save_dir}/{pred_args.name}"
    print(pred_args.exp_dir)

    # Create model directory
    if not os.path.isdir(pred_args.exp_dir):
        os.makedirs(pred_args.exp_dir)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    with open(pred_args.exp_dir + "/args.txt", "w") as f:
        json.dump(pred_args.__dict__, f, indent=2)
    # Automatically choose GPU if available
    pred_args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print("Args:", pred_args)

    # Where the magic is
    main(pred_args)
