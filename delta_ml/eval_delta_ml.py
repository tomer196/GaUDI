import argparse
import os
import json
import random
import shutil
from datetime import datetime
from time import time, sleep
import warnings

import sys, os

from torch.nn.functional import l1_loss

from delta_ml.train_delta_ml import (
    DeltaMLArgs,
    create_data_loaders,
    get_predictor_model,
    compute_loss,
    plot_correlations,
)

sys.path.append("/home/tomerweiss/PBHs-design")

import matplotlib.pyplot as plt
from edm.equivariant_diffusion.utils import (
    remove_mean_with_mask,
    assert_correctly_masked,
    assert_mean_zero_with_mask,
)
import numpy as np
import pandas as pd
import torch

from train_cond_predictor import check_mask_correct

warnings.simplefilter(action="ignore", category=FutureWarning)


def predict(model, x, h, node_mask, edge_mask):
    xh = torch.cat([x, h], dim=2)

    bs, n_nodes, n_dims = x.size()
    assert_correctly_masked(x, node_mask)
    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
    pred = model(xh, node_mask, edge_mask)
    return pred


def val_epoch(
    tag,
    predictor,
    dataloader,
    args,
):
    predictor.eval()
    with torch.no_grad():
        start_time = time()
        loss_list = []
        rl_loss = []
        pred_list = []
        targets_list = []
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

            pred = predict(
                predictor,
                x,
                h,
                node_mask,
                edge_mask,
            )
            loss = l1_loss(pred, y)

            pred_list.append(pred.cpu())
            targets_list.append(y.cpu())
            loss_list.append(loss.item())
            rl_loss.append(dataloader.dataset.rescale_loss(loss).item())

            # tepoch.set_postfix(loss=np.mean(loss_list).item())
        print(
            f"[{tag}] loss: {np.mean(loss_list):.4f}+-{np.std(loss_list):.4f}, "
            f"L1 (rescaled): {np.mean(rl_loss):.4f}, "
            f" in {int(time() - start_time)} secs"
        )
        sleep(0.01)
    pred = (
        torch.cat(pred_list) * dataloader.dataset.target_std
        + dataloader.dataset.target_mean
    )
    targets = (
        torch.cat(targets_list) * dataloader.dataset.target_std
        + dataloader.dataset.target_mean
    )
    return np.mean(loss_list), pred, targets


def plot_correlations_prediction(y, x, tag, target_features, idxs=None):
    if idxs is None:
        idxs = list(range(len(target_features)))
    for i in idxs:
        plt.scatter(x[:, i], y[:, i])
        plt.title(f"{tag} - {target_features[i]}")
        plt.xlabel("Prediction")
        plt.ylabel("DFT")
        plt.show()


@torch.no_grad()
def main(pred_args):
    # Prepare data
    train_loader, val_loader, test_loader = create_data_loaders(pred_args)
    plot_correlations(train_loader.dataset)
    predictor = get_predictor_model(pred_args, train_loader.dataset)
    target_features = train_loader.dataset.target_features
    idxs = [1]

    print("Begin evaluating")
    loss, pred, targets = val_epoch("train", predictor, train_loader, pred_args)
    plot_correlations_prediction(pred, targets, "train", target_features, idxs)
    loss, pred, targets = val_epoch("val", predictor, val_loader, pred_args)
    plot_correlations_prediction(pred, targets, "val", target_features, idxs)
    loss, pred, targets = val_epoch("test", predictor, test_loader, pred_args)
    plot_correlations_prediction(pred, targets, "test", target_features, idxs)


if __name__ == "__main__":
    pred_args = DeltaMLArgs().parse_args()
    pred_args.name = "hetro2dft"
    pred_args.exp_dir = f"{pred_args.save_dir}/{pred_args.name}"
    print(pred_args.exp_dir)

    # Create model directory
    if not os.path.isdir(pred_args.exp_dir):
        os.makedirs(pred_args.exp_dir)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    with open(pred_args.exp_dir + "/args.txt", "r") as f:
        pred_args.__dict__ = json.load(f)
    pred_args.restore = True
    # Automatically choose GPU if available
    pred_args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print("Args:", pred_args)

    # Where the magic is
    main(pred_args)
